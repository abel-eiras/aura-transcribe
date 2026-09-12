"""Etapa de diarización con pyannote.

Se usa pyannote directamente (en vez del envoltorio de WhisperX) para poder
controlar el modelo, los límites de hablantes y la degradación a CPU si los
6 GB de VRAM se quedan cortos.

Requiere un token de Hugging Face con las condiciones de
`pyannote/speaker-diarization-community-1` y `pyannote/segmentation-3.0`
aceptadas. Si no hay token se lanza ErrorConfig con instrucciones.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .asr import liberar_memoria
from .config import Config, exigir_token_hf
from .merge import Turno

log = logging.getLogger("aura.diarize")


class ErrorDiarizacion(RuntimeError):
    """La diarización no pudo completarse."""


def _turnos_desde_salida(salida) -> list[Turno]:
    """Normaliza la salida de pyannote a una lista de Turno.

    pyannote 3.x devuelve directamente un `Annotation`; los pipelines de la
    familia community-1 (pyannote 4.x) devuelven un objeto con el `Annotation`
    dentro (`.speaker_diarization`), así que se contemplan ambos casos.
    """
    anotacion = salida
    for atributo in ("speaker_diarization", "diarization", "annotation"):
        if hasattr(anotacion, atributo):
            anotacion = getattr(anotacion, atributo)
            break

    if not hasattr(anotacion, "itertracks"):
        raise ErrorDiarizacion(
            f"Salida de diarización no reconocida: {type(salida).__name__}"
        )

    turnos = [
        Turno(inicio=float(segmento.start), fin=float(segmento.end), hablante=str(etiqueta))
        for segmento, _, etiqueta in anotacion.itertracks(yield_label=True)
    ]
    turnos.sort(key=lambda t: (t.inicio, t.fin))
    return turnos


def _cargar_pipeline(cfg: Config, token: str, dispositivo: str):
    from pyannote.audio import Pipeline

    log.info("Cargando modelo de diarización %s en %s", cfg.diarizacion.modelo, dispositivo)
    try:
        pipeline = Pipeline.from_pretrained(cfg.diarizacion.modelo, token=token)
    except TypeError:
        # Versiones antiguas de pyannote usan use_auth_token en vez de token.
        pipeline = Pipeline.from_pretrained(
            cfg.diarizacion.modelo, use_auth_token=token
        )

    if pipeline is None:
        raise ErrorDiarizacion(
            f"No se pudo cargar {cfg.diarizacion.modelo}. Lo más probable es que "
            "falte aceptar las condiciones del modelo en huggingface.co con la "
            "cuenta del token, o que el token no tenga permiso de lectura."
        )

    try:
        import torch

        pipeline.to(torch.device(dispositivo))
    except Exception as exc:
        log.warning("No se pudo mover el pipeline a %s (%s); se usa CPU", dispositivo, exc)

    return pipeline


def _entrada_pyannote(ruta_wav: Path):
    """Prepara la entrada de audio en memoria para pyannote.

    pyannote 4.x lee ficheros con torchcodec, que solo soporta ffmpeg 4-7; en
    esta máquina el ffmpeg del sistema es la 8, así que se le pasa el audio ya
    cargado como {'waveform': tensor, 'sample_rate': int} y se evita el problema.
    """
    import torch

    from .audio import cargar_wav_mono

    muestras, frecuencia = cargar_wav_mono(ruta_wav)
    forma_onda = torch.from_numpy(muestras).unsqueeze(0)  # (canal, tiempo)
    return {"waveform": forma_onda, "sample_rate": frecuencia}


def _ejecutar(cfg: Config, pipeline, ruta_wav: Path) -> list[Turno]:
    parametros: dict = {}
    if cfg.diarizacion.max_hablantes and cfg.diarizacion.max_hablantes > 0:
        parametros["max_speakers"] = cfg.diarizacion.max_hablantes
    if cfg.diarizacion.min_hablantes and cfg.diarizacion.min_hablantes > 0:
        parametros["min_speakers"] = cfg.diarizacion.min_hablantes

    entrada = _entrada_pyannote(ruta_wav)

    try:
        salida = pipeline(entrada, **parametros)
    except TypeError as exc:
        # Algún pipeline puede no aceptar min/max_speakers: se reintenta sin ellos.
        log.warning("El pipeline no acepta límites de hablantes (%s); se ejecuta sin ellos", exc)
        salida = pipeline(entrada)

    return _turnos_desde_salida(salida)


def diarizar(cfg: Config, ruta_wav: Path) -> list[Turno]:
    """Devuelve los turnos de habla del WAV normalizado.

    Lanza ErrorConfig (con instrucciones) si falta el token de Hugging Face.
    """
    token = exigir_token_hf()  # Nunca se registra ni se imprime.

    dispositivo = cfg.asr.dispositivo
    try:
        import torch

        if dispositivo == "cuda" and not torch.cuda.is_available():
            dispositivo = "cpu"
    except Exception:
        dispositivo = "cpu"

    pipeline = None
    try:
        pipeline = _cargar_pipeline(cfg, token, dispositivo)
        turnos = _ejecutar(cfg, pipeline, ruta_wav)
    except (RuntimeError, MemoryError) as exc:
        mensaje = str(exc).lower()
        sin_memoria = "out of memory" in mensaje or "cuda" in mensaje
        if not (sin_memoria and cfg.diarizacion.reserva_cpu and dispositivo == "cuda"):
            raise
        log.warning("La diarización se quedó sin VRAM; se reintenta en CPU (más lento)")
        liberar_memoria(pipeline)
        pipeline = _cargar_pipeline(cfg, token, "cpu")
        turnos = _ejecutar(cfg, pipeline, ruta_wav)
    finally:
        liberar_memoria(pipeline)

    hablantes = {t.hablante for t in turnos}
    log.info("Diarización: %d turnos, %d hablantes", len(turnos), len(hablantes))
    return turnos
