"""Utilidades de audio: metadatos con ffprobe y normalización con ffmpeg."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

log = logging.getLogger("aura.audio")


class ErrorAudio(RuntimeError):
    """ffmpeg/ffprobe falló o el fichero no es audio válido."""


def duracion_segundos(ruta: Path) -> float:
    """Duración del audio en segundos según ffprobe."""
    orden = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(ruta),
    ]
    try:
        salida = subprocess.run(
            orden, capture_output=True, text=True, check=True, timeout=120
        ).stdout
    except FileNotFoundError as exc:
        raise ErrorAudio("No se encontró ffprobe en el PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise ErrorAudio(f"ffprobe falló sobre {ruta.name}: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ErrorAudio(f"ffprobe se colgó sobre {ruta.name}") from exc

    try:
        return float(json.loads(salida)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ErrorAudio(f"ffprobe no devolvió duración para {ruta.name}") from exc


def normalizar(origen: Path, destino: Path, frecuencia: int = 16000) -> Path:
    """Convierte el audio a WAV PCM 16 bits mono a `frecuencia` Hz.

    Siempre se escribe en el scratch local, nunca dentro de la carpeta de trabajo sincronizada.
    """
    origen = Path(origen)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    orden = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(origen),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(frecuencia),
        "-c:a",
        "pcm_s16le",
        str(destino),
    ]
    log.debug("Normalizando %s -> %s", origen.name, destino)
    try:
        subprocess.run(orden, capture_output=True, text=True, check=True, timeout=3600)
    except FileNotFoundError as exc:
        raise ErrorAudio("No se encontró ffmpeg en el PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise ErrorAudio(
            f"ffmpeg no pudo convertir {origen.name}: {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ErrorAudio(f"ffmpeg se colgó convirtiendo {origen.name}") from exc

    if not destino.is_file() or destino.stat().st_size == 0:
        raise ErrorAudio(f"La conversión de {origen.name} produjo un WAV vacío")
    return destino


def cargar_wav_mono(ruta: Path) -> tuple["object", int]:
    """Lee un WAV PCM 16 bits mono y lo devuelve como (float32 en [-1,1], frecuencia).

    Se usa la librería estándar `wave` a propósito: el ffmpeg del sistema es la
    versión 8 y torchcodec (que usa pyannote 4.x para leer audio) solo soporta
    hasta la 7, así que se evita ese camino por completo cargando el audio a
    mano. El WAV siempre lo genera antes `normalizar()`, con formato conocido.
    """
    import wave

    import numpy as np

    with wave.open(str(ruta), "rb") as fh:
        canales = fh.getnchannels()
        ancho = fh.getsampwidth()
        frecuencia = fh.getframerate()
        crudo = fh.readframes(fh.getnframes())

    if ancho != 2:
        raise ErrorAudio(
            f"{ruta.name}: se esperaba PCM de 16 bits y tiene {ancho * 8} bits"
        )

    muestras = np.frombuffer(crudo, dtype="<i2").astype(np.float32) / 32768.0
    if canales > 1:
        muestras = muestras.reshape(-1, canales).mean(axis=1)
    return muestras, frecuencia
