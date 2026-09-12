"""Descubrimiento de los audios pendientes de transcribir.

Un audio está pendiente si:
  * está en la raíz de la carpeta de trabajo (no en processed/ ni transcription/),
  * su extensión está en la lista de extensiones de audio (el JPEG se ignora),
  * no existe ya un JSON con su nombre en transcription/ ni en transcription/sent/,
  * es estable (tamaño/mtime sin cambios en los últimos N segundos).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .filesystem import es_estable


@dataclass(frozen=True)
class AudioPendiente:
    ruta: Path
    tamano: int
    mtime: float

    @property
    def nombre(self) -> str:
        return self.ruta.name

    @property
    def nombre_base(self) -> str:
        return self.ruta.stem


def ruta_json_salida(cfg: Config, audio: Path) -> Path:
    """Ruta del JSON que produce este audio."""
    return cfg.rutas.transcripciones / f"{Path(audio).stem}.json"


def ya_transcrito(cfg: Config, audio: Path) -> bool:
    """True si ya existe el JSON en transcription/ o en transcription/sent/."""
    nombre = f"{Path(audio).stem}.json"
    return (cfg.rutas.transcripciones / nombre).exists() or (
        cfg.rutas.enviados / nombre
    ).exists()


def _candidatos(cfg: Config) -> list[Path]:
    """Ficheros de la raíz cuya extensión es de audio, ordenados por nombre."""
    base = cfg.rutas.base
    if not base.is_dir():
        return []
    extensiones = set(cfg.audio.extensiones)
    resultado = [
        hijo
        for hijo in base.iterdir()
        if hijo.is_file() and hijo.suffix.lower() in extensiones
    ]
    return sorted(resultado, key=lambda p: p.name)


def descubrir_pendientes(
    cfg: Config,
    *,
    forzar: bool = False,
    ignorar_estabilidad: bool = False,
) -> tuple[list[AudioPendiente], list[tuple[Path, str]]]:
    """Devuelve (pendientes, descartados) donde descartados es (ruta, motivo)."""
    pendientes: list[AudioPendiente] = []
    descartados: list[tuple[Path, str]] = []

    for ruta in _candidatos(cfg):
        if not forzar and ya_transcrito(cfg, ruta):
            descartados.append((ruta, "ya tiene JSON"))
            continue
        if not ignorar_estabilidad and not es_estable(
            ruta, cfg.audio.segundos_estabilidad
        ):
            descartados.append(
                (ruta, f"inestable (<{cfg.audio.segundos_estabilidad:.0f}s desde el último cambio)")
            )
            continue
        try:
            st = ruta.stat()
        except OSError as exc:
            descartados.append((ruta, f"no se puede leer: {exc}"))
            continue
        pendientes.append(AudioPendiente(ruta=ruta, tamano=st.st_size, mtime=st.st_mtime))

    return pendientes, descartados


def resolver_audio_suelto(cfg: Config, ruta: Path) -> AudioPendiente:
    """Prepara un AudioPendiente para un fichero concreto (--file)."""
    ruta = Path(ruta).expanduser()
    if not ruta.is_absolute():
        candidato = cfg.rutas.base / ruta
        ruta = candidato if candidato.exists() else ruta.resolve()
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el audio {ruta}")
    st = ruta.stat()
    return AudioPendiente(ruta=ruta, tamano=st.st_size, mtime=st.st_mtime)
