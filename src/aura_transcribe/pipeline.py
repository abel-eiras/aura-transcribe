"""Orquestación: de un audio en la carpeta de trabajo a su JSON en transcription/.

Orden estricto (sección 5 del plan):
  guardas -> descubrimiento -> normalización -> ASR -> alineamiento ->
  diarización -> fusión y limpieza -> JSON atómico -> mover el audio.

El audio SOLO se mueve a processed/ cuando el JSON ya está en disco.
Ante cualquier error el audio se queda donde está y se anota un reintento.

Por defecto cada audio se procesa en su PROPIO subproceso (ver ConfigEjecucion):
la VRAM de la GTX 1660 se fragmenta a lo largo de un proceso de vida larga y la
única forma fiable de devolverla entera es que el proceso muera. El padre
conserva el flock, la contabilidad de reintentos y el resumen; el hijo solo hace
el trabajo pesado de un único audio.

Hay DOS locks distintos y ninguno sustituye al otro:
  * `cfg.rutas.fichero_lock`: lock propio del pipeline, lo coge el PADRE durante
    toda la pasada para que dos ejecuciones (timer + manual) no se pisen.
  * `cfg.rutas.lock_sincronizacion`: el lock de rclone, lo coge QUIEN MUTA (o sea
    el hijo, dentro de procesar_audio) y solo durante la ventana de escritura del
    JSON + movimiento del audio.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .audio import duracion_segundos, normalizar
from .config import RUTA_CONFIG_POR_DEFECTO, Config
from .discovery import AudioPendiente, descubrir_pendientes, resolver_audio_suelto, ruta_json_salida
from .filesystem import (
    ErrorLock,
    asegurar_directorios,
    escribir_texto_atomico,
    exigir_montaje,
    guardar_estado,
    leer_estado,
    lock_exclusivo,
    lock_sincronizacion,
    mover_atomico,
)
from .merge import asignar_hablantes, limpiar
from .output import construir_documento, escribir_salida

log = logging.getLogger("aura.pipeline")

# Prefijo de la única línea de STDOUT del worker que el padre interpreta.
# Todo lo demás que el hijo escriba por stdout se ignora sin más.
CENTINELA_RESULTADO = "@@AURA-RESULTADO@@"


class ErrorSubproceso(RuntimeError):
    """El subproceso que procesaba un audio no terminó bien.

    Se trata como cualquier otro fallo del pipeline: el audio se queda donde
    está, se anota el reintento y la pasada continúa con el siguiente.
    """


@dataclass
class Opciones:
    """Opciones de una ejecución (vienen de la CLI)."""

    fichero: Path | None = None
    forzar: bool = False
    sin_diarizacion: bool = False
    idioma: str | None = None
    conservar_audio: bool = False
    dry_run: bool = False
    limite: int | None = None
    # Ruta del config.toml en uso; se le pasa tal cual al subproceso para que
    # padre e hijo lean exactamente la misma configuración.
    ruta_config: Path | None = None
    # --sin-aislamiento: procesa en el propio proceso (depuración).
    sin_aislamiento: bool = False
    verboso: bool = False


@dataclass
class ResultadoAudio:
    audio: Path
    ok: bool
    json_salida: Path | None = None
    error: str | None = None
    segundos: float = 0.0
    hablantes: int = 0
    segmentos: int = 0


@dataclass
class ResultadoEjecucion:
    procesados: list[ResultadoAudio] = field(default_factory=list)
    fallidos: list[ResultadoAudio] = field(default_factory=list)
    omitidos: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def hubo_fallos(self) -> bool:
        return bool(self.fallidos)


def _scratch_temporal(cfg: Config) -> Path:
    cfg.rutas.scratch.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="aura-", dir=str(cfg.rutas.scratch)))


def procesar_audio(
    cfg: Config, pendiente: AudioPendiente, opciones: Opciones
) -> ResultadoAudio:
    """Procesa un único audio de principio a fin."""
    from .asr import transcribir  # import perezoso: cargar torch tarda segundos

    origen = pendiente.ruta
    inicio = time.perf_counter()
    log.info("=== %s (%.1f MB) ===", origen.name, pendiente.tamano / 1e6)

    trabajo = _scratch_temporal(cfg)
    try:
        duracion = duracion_segundos(origen)
        log.info("Duración: %.2f s", duracion)

        wav = normalizar(
            origen, trabajo / f"{origen.stem}.16k.wav", cfg.audio.frecuencia_destino
        )

        resultado_asr = transcribir(
            cfg, wav, idioma_cli=opciones.idioma, ruta_original=origen
        )
        segmentos = resultado_asr.segmentos
        log.info("ASR: %d segmentos", len(segmentos))

        diarizacion_activa = cfg.diarizacion.activa and not opciones.sin_diarizacion
        if diarizacion_activa and segmentos:
            from .diarize import diarizar

            turnos = diarizar(cfg, wav)
            segmentos = asignar_hablantes(
                segmentos, turnos, usar_palabras=resultado_asr.hubo_alineamiento
            )
        elif not diarizacion_activa:
            log.info("Diarización desactivada: todos los segmentos van a SPEAKER_00")
            for seg in segmentos:
                seg.hablante = "SPEAKER_00"

        segmentos = limpiar(
            segmentos,
            max_hablantes=cfg.diarizacion.max_hablantes,
            hueco_fusion_ms=cfg.limpieza.hueco_fusion_ms,
            duracion_minima_s=cfg.limpieza.duracion_minima_s,
        )
        log.info("Tras la limpieza: %d segmentos", len(segmentos))

        segundos_proceso = time.perf_counter() - inicio

        documento = construir_documento(
            cfg=cfg,
            nombre_origen=origen.name,
            duracion_segundos=duracion,
            idioma_detectado=resultado_asr.idioma_detectado,
            confianza_idioma=resultado_asr.confianza_idioma,
            idioma_usado=resultado_asr.idioma_usado,
            segmentos=segmentos,
            hubo_alineamiento=resultado_asr.hubo_alineamiento,
            diarizacion_activa=diarizacion_activa,
            segundos_proceso=segundos_proceso,
        )

        destino_json = ruta_json_salida(cfg, origen)

        # --- Ventana de mutación de la carpeta de trabajo -------------------
        # Todo lo anterior (normalizar, ASR, diarización) trabaja en el scratch
        # local y no toca la carpeta de trabajo: puede durar 20 minutos sin molestar a nadie.
        # Lo que viene ahora sí muta el árbol sincronizado, así que se hace bajo
        # el MISMO flock que usa rclone bisync. Son milisegundos, y es lo que
        # impide que bisync fotografíe un estado a medias (JSON escrito pero
        # audio aún en la raíz, o al revés).
        with lock_sincronizacion(
            cfg.rutas.lock_sincronizacion,
            espera_maxima=cfg.ejecucion.espera_lock_sincronizacion,
        ):
            escribir_salida(cfg, documento, destino_json)
            log.info("JSON escrito en %s", destino_json)

            # Solo ahora, con el JSON ya en disco, se mueve el audio.
            if not opciones.conservar_audio and origen.parent == cfg.rutas.base:
                destino_audio = mover_atomico(origen, cfg.rutas.procesados)
                log.info("Audio movido a %s", destino_audio)
            elif opciones.conservar_audio:
                log.info("--keep-audio: el audio se queda en su sitio")
        # --- Fin de la ventana de mutación --------------------------------

        return ResultadoAudio(
            audio=origen,
            ok=True,
            json_salida=destino_json,
            segundos=segundos_proceso,
            hablantes=documento["metadata"]["num_speakers_detected"],
            segmentos=len(documento["segments"]),
        )
    finally:
        shutil.rmtree(trabajo, ignore_errors=True)


# ---------------------------------------------------------------------------
# Aislamiento en subproceso
# ---------------------------------------------------------------------------


def opciones_a_dict(opciones: Opciones) -> dict:
    """Serializa las Opciones a algo que JSON entienda (Path -> str)."""
    datos = asdict(opciones)
    for clave in ("fichero", "ruta_config"):
        datos[clave] = None if datos[clave] is None else str(datos[clave])
    # El hijo NUNCA vuelve a aislar: llama a procesar_audio() directamente.
    # Se marca de todos modos para que la opción no mienta si alguien la mira.
    datos["sin_aislamiento"] = True
    return datos


def opciones_desde_dict(datos: dict) -> Opciones:
    """Inverso de opciones_a_dict(); ignora claves desconocidas."""
    campos = {f: datos[f] for f in Opciones.__dataclass_fields__ if f in datos}
    for clave in ("fichero", "ruta_config"):
        if campos.get(clave) is not None:
            campos[clave] = Path(campos[clave])
    return Opciones(**campos)


def resultado_a_dict(resultado: ResultadoAudio) -> dict:
    """Serializa un ResultadoAudio (Path -> str)."""
    datos = asdict(resultado)
    datos["audio"] = str(datos["audio"])
    datos["json_salida"] = (
        None if datos["json_salida"] is None else str(datos["json_salida"])
    )
    return datos


def resultado_desde_dict(datos: dict) -> ResultadoAudio:
    """Inverso de resultado_a_dict()."""
    return ResultadoAudio(
        audio=Path(datos["audio"]),
        ok=bool(datos.get("ok", False)),
        json_salida=(
            Path(datos["json_salida"]) if datos.get("json_salida") else None
        ),
        error=datos.get("error"),
        segundos=float(datos.get("segundos", 0.0)),
        hablantes=int(datos.get("hablantes", 0)),
        segmentos=int(datos.get("segmentos", 0)),
    )


def ruta_config_efectiva(opciones: Opciones) -> Path:
    """El config.toml que debe leer el hijo, resuelto igual que lo hace la CLI."""
    if opciones.ruta_config is not None:
        return Path(opciones.ruta_config)
    return Path(os.environ.get("AURA_CONFIG", str(RUTA_CONFIG_POR_DEFECTO)))


def _orden_worker(pendiente: AudioPendiente, opciones: Opciones) -> list[str]:
    """La orden exacta con la que se lanza el worker."""
    orden = [
        sys.executable,
        "-m",
        "aura_transcribe.worker",
        "--config",
        str(ruta_config_efectiva(opciones)),
        "--audio",
        str(pendiente.ruta),
        "--opciones",
        json.dumps(opciones_a_dict(opciones), ensure_ascii=False),
    ]
    if opciones.verboso:
        orden.append("--verboso")
    return orden


def _entorno_worker() -> dict:
    """Entorno del hijo: el del padre + el src/ de este mismo paquete en PYTHONPATH.

    Así el hijo importa exactamente el mismo código que el padre aunque el
    paquete no esté instalado en el intérprete (ejecución desde el árbol fuente).
    """
    entorno = dict(os.environ)
    raiz_paquete = str(Path(__file__).resolve().parents[1])
    anterior = entorno.get("PYTHONPATH", "")
    if raiz_paquete not in anterior.split(os.pathsep):
        entorno["PYTHONPATH"] = (
            f"{raiz_paquete}{os.pathsep}{anterior}" if anterior else raiz_paquete
        )
    return entorno


def _extraer_carga(salida: str | None) -> dict | None:
    """Devuelve el JSON de la línea centinela, o None si no hay ninguna válida."""
    for linea in reversed((salida or "").splitlines()):
        linea = linea.strip()
        if not linea.startswith(CENTINELA_RESULTADO):
            continue
        try:
            carga = json.loads(linea[len(CENTINELA_RESULTADO) :].strip())
        except json.JSONDecodeError:
            return None
        return carga if isinstance(carga, dict) else None
    return None


def _motivo_salida(codigo: int) -> str:
    """Explica en cristiano por qué murió el hijo."""
    if codigo < 0:
        try:
            nombre = signal.Signals(-codigo).name
        except ValueError:
            nombre = f"señal {-codigo}"
        extra = (
            " (probablemente el OOM killer del sistema: se quedó sin RAM)"
            if -codigo == signal.SIGKILL
            else ""
        )
        return f"lo mató {nombre}{extra}"
    return f"salió con código {codigo}"


def _procesar_en_subproceso(
    cfg: Config, pendiente: AudioPendiente, opciones: Opciones
) -> ResultadoAudio:
    """Procesa un audio en un proceso aparte y devuelve su ResultadoAudio.

    Solo se captura STDOUT (por donde viaja el JSON del resultado). STDERR se
    hereda a propósito: los logs del hijo salen en vivo por consola y el propio
    hijo los añade también al fichero de log.

    El hijo NO coge el flock: llama directamente a procesar_audio(), nunca a
    ejecutar_con_lock(). El lock del padre sigue siendo el único.
    """
    orden = _orden_worker(pendiente, opciones)
    log.info("Lanzando subproceso para %s (pid padre %d)", pendiente.nombre, os.getpid())
    log.debug("Orden: %s", orden)

    proceso = subprocess.run(  # noqa: S603 - orden construida por nosotros
        orden,
        stdout=subprocess.PIPE,
        stderr=None,  # heredado: los logs del hijo salen en vivo
        text=True,
        env=_entorno_worker(),
        check=False,
    )

    carga = _extraer_carga(proceso.stdout)

    if carga is None:
        raise ErrorSubproceso(
            f"El subproceso de {pendiente.nombre} {_motivo_salida(proceso.returncode)} "
            "y no dejó ningún resultado en su salida estándar. "
            "Mira el log para ver hasta dónde llegó."
        )

    if not carga.get("ok"):
        detalle = carga.get("traceback") or carga.get("error") or "sin detalle"
        raise ErrorSubproceso(
            f"El subproceso de {pendiente.nombre} falló "
            f"({_motivo_salida(proceso.returncode)}):\n{detalle}"
        )

    return resultado_desde_dict(carga)


def _procesar(
    cfg: Config, pendiente: AudioPendiente, opciones: Opciones
) -> ResultadoAudio:
    """Procesa un audio, aislado o no según la config y la CLI."""
    if cfg.ejecucion.aislar_subproceso and not opciones.sin_aislamiento:
        return _procesar_en_subproceso(cfg, pendiente, opciones)
    log.info("Sin aislamiento: %s se procesa en este mismo proceso", pendiente.nombre)
    return procesar_audio(cfg, pendiente, opciones)


def _registrar_fallo(
    cfg: Config, estado: dict, audio: Path, error: str
) -> None:
    """Anota el reintento y, si se agotaron, mueve el audio a failed/."""
    entrada = estado.get(audio.name, {})
    intentos = int(entrada.get("intentos", 0)) + 1
    entrada.update(
        {
            "intentos": intentos,
            "ultimo_error": error.strip().splitlines()[-1][:500] if error.strip() else "",
            "ultimo_intento": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    estado[audio.name] = entrada

    if intentos >= cfg.errores.intentos_maximos and audio.exists():
        asegurar_directorios(cfg.rutas.fallidos)
        destino = mover_atomico(audio, cfg.rutas.fallidos)
        escribir_texto_atomico(
            destino.with_suffix(destino.suffix + ".error.txt"),
            f"Intentos: {intentos}\nFecha: {entrada['ultimo_intento']}\n\n{error}\n",
        )
        entrada["movido_a_fallidos"] = str(destino)
        log.error(
            "%s falló %d veces: movido a %s (no se borra nada)",
            audio.name,
            intentos,
            destino,
        )
    else:
        log.error(
            "%s falló (intento %d/%d); el audio se queda donde está",
            audio.name,
            intentos,
            cfg.errores.intentos_maximos,
        )


def ejecutar(cfg: Config, opciones: Opciones) -> ResultadoEjecucion:
    """Ejecuta una pasada completa sobre el backlog (o sobre --file)."""
    resultado = ResultadoEjecucion()

    if cfg.rutas.punto_montaje is not None:
        exigir_montaje(cfg.rutas.punto_montaje)
    if not cfg.rutas.base.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de trabajo {cfg.rutas.base}")

    if opciones.fichero is not None:
        pendientes = [resolver_audio_suelto(cfg, opciones.fichero)]
        descartados: list[tuple[Path, str]] = []
        if not opciones.forzar:
            from .discovery import ya_transcrito

            if ya_transcrito(cfg, pendientes[0].ruta):
                log.warning(
                    "%s ya tiene JSON; usa --force para rehacerlo", pendientes[0].nombre
                )
                resultado.omitidos.append((pendientes[0].ruta, "ya tiene JSON"))
                return resultado
    else:
        pendientes, descartados = descubrir_pendientes(cfg, forzar=opciones.forzar)

    resultado.omitidos.extend(descartados)

    if opciones.limite:
        pendientes = pendientes[: opciones.limite]

    if opciones.dry_run:
        log.info("--dry-run: no se procesa ni se mueve nada")
        log.info("Pendientes (%d):", len(pendientes))
        for pendiente in pendientes:
            log.info(
                "  + %-32s %8.1f MB -> %s",
                pendiente.nombre,
                pendiente.tamano / 1e6,
                ruta_json_salida(cfg, pendiente.ruta).name,
            )
        if descartados:
            log.info("Ignorados (%d):", len(descartados))
            for ruta, motivo in descartados:
                log.info("  - %-32s %s", ruta.name, motivo)
        resultado.procesados = [
            ResultadoAudio(audio=p.ruta, ok=True) for p in pendientes
        ]
        return resultado

    if not pendientes:
        log.info("No hay audios pendientes. Nada que hacer.")
        return resultado

    asegurar_directorios(
        cfg.rutas.transcripciones, cfg.rutas.procesados, cfg.rutas.scratch
    )

    estado = leer_estado(cfg.rutas.fichero_estado)

    for pendiente in pendientes:
        try:
            parcial = _procesar(cfg, pendiente, opciones)
            resultado.procesados.append(parcial)
            estado.pop(pendiente.nombre, None)
            log.info(
                "OK %s en %.1f s (%d segmentos, %d hablantes)",
                pendiente.nombre,
                parcial.segundos,
                parcial.segmentos,
                parcial.hablantes,
            )
        except KeyboardInterrupt:
            log.warning("Interrumpido por el usuario; el audio se queda intacto")
            raise
        except Exception as exc:
            detalle = traceback.format_exc()
            log.error("Error con %s: %s", pendiente.nombre, exc)
            log.debug(detalle)
            _registrar_fallo(cfg, estado, pendiente.ruta, detalle)
            resultado.fallidos.append(
                ResultadoAudio(audio=pendiente.ruta, ok=False, error=str(exc))
            )
        finally:
            guardar_estado(cfg.rutas.fichero_estado, estado)

    return resultado


def ejecutar_con_lock(cfg: Config, opciones: Opciones) -> ResultadoEjecucion:
    """Igual que ejecutar() pero protegido por flock para que no se pisen dos pasadas."""
    try:
        with lock_exclusivo(cfg.rutas.fichero_lock):
            return ejecutar(cfg, opciones)
    except ErrorLock as exc:
        log.warning("%s", exc)
        raise
