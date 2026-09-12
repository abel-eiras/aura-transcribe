"""Operaciones de sistema de ficheros seguras: montaje, lock, escrituras atómicas.

Reglas de oro del pipeline:
  * Nunca se borra nada del usuario.
  * El JSON se escribe primero en un temporal del MISMO sistema de ficheros y
    luego se publica con os.replace() (atómico).
  * El audio solo se mueve DESPUÉS de que el JSON esté en disco.
  * La carpeta de trabajo la comparte rclone: la mutación se hace bajo el mismo
    flock que usa el script de sincronización (ver lock_sincronizacion()).
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("aura.filesystem")


class ErrorMontaje(RuntimeError):
    """El volumen de datos no está montado."""


class ErrorLock(RuntimeError):
    """Ya hay otra ejecución del pipeline en marcha."""


class ErrorLockSincronizacion(RuntimeError):
    """No se pudo coordinar con rclone dentro del plazo previsto.

    NO es hermana de ErrorLock a propósito: aquella significa "sale sin hacer
    nada, ya hay otra pasada" (código 2 de la CLI), mientras que esta es un
    fallo normal de UN audio: el fichero se queda donde está y se anota el
    reintento como con cualquier otro error.
    """


def esta_montado(punto: Path) -> bool:
    """True si `punto` es un punto de montaje real (o la raíz de uno)."""
    punto = Path(punto)
    if not punto.exists():
        return False
    if os.path.ismount(punto):
        return True
    # Comprobación de reserva: el dispositivo difiere del del padre.
    try:
        padre = punto.parent
        return punto.stat().st_dev != padre.stat().st_dev
    except OSError:
        return False


def exigir_montaje(punto: Path) -> None:
    """Lanza ErrorMontaje si el volumen no está montado."""
    if not esta_montado(punto):
        raise ErrorMontaje(
            f"El volumen {punto} no está montado; no se toca nada. "
            "Monta la unidad y vuelve a lanzar el pipeline."
        )


@contextmanager
def lock_exclusivo(ruta_lock: Path, bloqueante: bool = False) -> Iterator[int]:
    """flock() exclusivo sobre `ruta_lock` para que cron y manual no se pisen."""
    ruta_lock = Path(ruta_lock)
    ruta_lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(ruta_lock), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        banderas = fcntl.LOCK_EX if bloqueante else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(fd, banderas)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                raise ErrorLock(
                    f"Ya hay otra ejecución en marcha (lock {ruta_lock}). Se sale sin hacer nada."
                ) from exc
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# Cada cuánto se reintenta coger el lock de sincronización mientras se espera.
INTERVALO_ESPERA_SINCRONIZACION = 0.2


@contextmanager
def lock_sincronizacion(
    ruta_lock: Path | None,
    espera_maxima: float = 120.0,
    intervalo: float = INTERVALO_ESPERA_SINCRONIZACION,
) -> Iterator[int | None]:
    """flock compartido con un sincronizador externo, solo durante la ventana de
    mutación de la carpeta de trabajo.

    Ejemplo de uso: la carpeta de trabajo está sincronizada BIDIRECCIONALMENTE
    con un servicio externo (Google Drive, Nextcloud...) por un script propio
    que hace `rclone bisync` cada 15 min protegido con `flock -n` sobre un
    fichero de lock. Si el pipeline escribe el JSON y renombra el audio justo
    mientras ese script está recorriendo el árbol, bisync puede ver un estado
    a medias.

    La solución es que ambos usen el MISMO lock, pero con dos matices:

      * El pipeline lo coge solo alrededor de la escritura del JSON y del
        `os.rename` del audio: milisegundos. Cogerlo durante toda la
        transcripción dejaría al usuario sin sincronizar 20 minutos.
      * Aquí sí se ESPERA (con plazo), al revés que en lock_exclusivo(): si
        rclone está a media pasada, lo que queremos es esperar a que acabe, no
        rendirnos.

    `ruta_lock` a None desactiva la coordinación (útil en tests y en máquinas
    sin rclone). Si el plazo se agota se lanza ErrorLockSincronizacion, que el
    pipeline trata como un fallo normal del audio: se queda donde está y se
    anota el reintento.
    """
    if ruta_lock is None:
        yield None
        return

    ruta_lock = Path(ruta_lock)
    ruta_lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(ruta_lock), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        limite = time.monotonic() + max(0.0, espera_maxima)
        avisado = False
        inicio = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
            if not avisado:
                log.info(
                    "Sincronización en curso (lock %s): espero hasta %.0f s antes "
                    "de tocar la carpeta de trabajo.",
                    ruta_lock,
                    espera_maxima,
                )
                avisado = True
            if time.monotonic() >= limite:
                raise ErrorLockSincronizacion(
                    f"No se pudo coger el lock de sincronización {ruta_lock} en "
                    f"{espera_maxima:.0f} s (rclone sigue trabajando). No se toca "
                    "nada: el audio se queda donde está y se reintentará."
                )
            time.sleep(min(intervalo, max(0.0, limite - time.monotonic())))

        if avisado:
            log.info(
                "Lock de sincronización conseguido tras esperar %.1f s.",
                time.monotonic() - inicio,
            )
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def asegurar_directorios(*rutas: Path) -> None:
    """Crea los directorios que falten (no toca los existentes)."""
    for ruta in rutas:
        Path(ruta).mkdir(parents=True, exist_ok=True)


def escribir_texto_atomico(destino: Path, contenido: str) -> None:
    """Escribe texto de forma atómica: temporal hermano + fsync + os.replace()."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_name(f".{destino.name}.tmp-{os.getpid()}")
    try:
        with temporal.open("w", encoding="utf-8") as fh:
            fh.write(contenido)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporal, destino)
    except BaseException:
        temporal.unlink(missing_ok=True)
        raise
    # Se sincroniza el directorio para que el rename sobreviva a un corte.
    try:
        fd_dir = os.open(str(destino.parent), os.O_RDONLY)
        try:
            os.fsync(fd_dir)
        finally:
            os.close(fd_dir)
    except OSError:
        pass  # Sistemas de ficheros que no lo permiten (p. ej. algunos FUSE).


def escribir_json_atomico(destino: Path, datos: Any) -> None:
    """Serializa `datos` a JSON legible y lo publica atómicamente."""
    texto = json.dumps(datos, ensure_ascii=False, indent=2) + "\n"
    escribir_texto_atomico(Path(destino), texto)


def mover_atomico(origen: Path, destino_dir: Path) -> Path:
    """Mueve `origen` a `destino_dir` con os.rename (atómico en el mismo volumen).

    Si el destino ya existe se añade un sufijo numérico: nunca se pisa un fichero
    del usuario. Si origen y destino están en volúmenes distintos se degrada a
    copiar + fsync + borrar el origen (shutil.move), que ya no es atómico.
    """
    origen = Path(origen)
    destino_dir = Path(destino_dir)
    destino_dir.mkdir(parents=True, exist_ok=True)

    destino = destino_dir / origen.name
    contador = 1
    while destino.exists():
        destino = destino_dir / f"{origen.stem}__{contador}{origen.suffix}"
        contador += 1

    try:
        os.rename(origen, destino)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.move(str(origen), str(destino))
    return destino


def es_estable(ruta: Path, segundos: float, ahora: float | None = None) -> bool:
    """True si el fichero lleva `segundos` sin cambiar de tamaño ni de mtime.

    Evita procesar una sincronización externa a medias. Se comprueba contra el
    reloj: si mtime es muy reciente, aún puede estar escribiéndose.
    """
    ruta = Path(ruta)
    try:
        st = ruta.stat()
    except OSError:
        return False
    if st.st_size == 0:
        return False
    referencia = time.time() if ahora is None else ahora
    return (referencia - st.st_mtime) >= segundos


def leer_estado(ruta: Path) -> dict:
    """Lee el fichero de estado (contadores de reintentos). Tolera corrupción."""
    ruta = Path(ruta)
    if not ruta.is_file():
        return {}
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return datos if isinstance(datos, dict) else {}


def guardar_estado(ruta: Path, estado: dict) -> None:
    """Guarda el estado de forma atómica."""
    escribir_json_atomico(Path(ruta), estado)
