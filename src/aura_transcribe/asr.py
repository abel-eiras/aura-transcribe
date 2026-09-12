"""Etapa de reconocimiento de voz con WhisperX (faster-whisper por debajo).

Incluye la regla de idioma restringido de `[idioma]` en config.toml:
  * la detección se hace sobre el primer tramo con VOZ (VAD), no sobre los
    primeros 30 s que pueden ser silencio;
  * la detección nunca es libre: solo `idioma.idioma_alternativo` (o uno de
    `idioma.confundibles_con_alternativo`, para idiomas próximos que el
    detector suele confundir) por encima del umbral cambia el idioma; en
    cualquier otro caso se usa `idioma.por_defecto`;
  * override manual por --language, por sufijo `_xx` en el nombre del fichero
    o por sidecar `<nombre>.lang`.
"""

from __future__ import annotations

import gc
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .output import Segmento

log = logging.getLogger("aura.asr")

_SUFIJO_IDIOMA = re.compile(r"_([a-z]{2})$", re.IGNORECASE)


@dataclass
class ResultadoAsr:
    segmentos: list[Segmento]
    idioma_detectado: str | None
    confianza_idioma: float | None
    idioma_usado: str
    hubo_alineamiento: bool


def liberar_memoria(*objetos) -> None:
    """Suelta modelos y vacía la caché de CUDA: las etapas van en serie en 6 GB."""
    for objeto in objetos:
        del objeto
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # torch puede no estar disponible en tests
        pass


def idioma_forzado(ruta_audio: Path, idioma_cli: str | None) -> str | None:
    """Override manual de idioma: CLI > sidecar .lang > sufijo _xx en el nombre."""
    if idioma_cli:
        return idioma_cli.strip().lower()

    sidecar = Path(ruta_audio).with_suffix(".lang")
    if sidecar.is_file():
        try:
            valor = sidecar.read_text(encoding="utf-8").strip().lower()
        except OSError:
            valor = ""
        if valor:
            log.info("Idioma forzado por sidecar %s: %s", sidecar.name, valor)
            return valor

    coincidencia = _SUFIJO_IDIOMA.search(Path(ruta_audio).stem)
    if coincidencia:
        valor = coincidencia.group(1).lower()
        log.info("Idioma forzado por sufijo del nombre: %s", valor)
        return valor

    return None


def decidir_idioma(
    cfg: Config, idioma_bruto: str | None, probabilidad: float | None
) -> str:
    """Aplica la regla de idioma restringido: detección nunca libre."""
    if idioma_bruto is None or probabilidad is None:
        return cfg.idioma.por_defecto

    alternativo = cfg.idioma.idioma_alternativo
    if (
        alternativo
        and idioma_bruto in cfg.idioma.confundibles_con_alternativo
        and probabilidad >= cfg.idioma.umbral_alternativo
        and alternativo in cfg.idioma.permitidos
    ):
        log.info(
            "Detección %s (%.2f) por encima del umbral: se transcribe en %s",
            idioma_bruto,
            probabilidad,
            alternativo,
        )
        return alternativo

    if idioma_bruto != cfg.idioma.por_defecto:
        log.info(
            "Detección %s (%.2f) descartada: se transcribe en %s (detección restringida)",
            idioma_bruto,
            probabilidad,
            cfg.idioma.por_defecto,
        )
    return cfg.idioma.por_defecto


def detectar_idioma(modelo, audio, cfg: Config) -> tuple[str | None, float | None]:
    """Detecta el idioma sobre el primer tramo con voz. Devuelve (idioma, prob)."""
    # faster-whisper analiza ventanas de 30 s; con VAD se salta el silencio inicial.
    ventanas = max(1, math.ceil(cfg.idioma.segundos_deteccion / 30.0))
    interno = getattr(modelo, "model", None)

    if interno is not None and hasattr(interno, "detect_language"):
        try:
            idioma, probabilidad, _ = interno.detect_language(
                audio=audio,
                vad_filter=True,
                language_detection_segments=ventanas,
            )
            return str(idioma), float(probabilidad)
        except Exception as exc:  # el VAD puede fallar en audios muy cortos
            log.warning("Detección de idioma con VAD fallida (%s); se reintenta sin VAD", exc)
            try:
                idioma, probabilidad, _ = interno.detect_language(audio=audio)
                return str(idioma), float(probabilidad)
            except Exception as exc2:
                log.warning("Detección de idioma no disponible: %s", exc2)

    # Reserva: la propia tubería de WhisperX (solo devuelve el código de idioma).
    try:
        idioma = modelo.detect_language(audio)
        return str(idioma), None
    except Exception as exc:
        log.warning("No se pudo detectar el idioma: %s", exc)
        return None, None


def cargar_modelo(cfg: Config, idioma: str | None = None):
    """Carga el modelo de WhisperX, degradando a CPU si CUDA no está disponible."""
    import whisperx

    dispositivo = cfg.asr.dispositivo
    tipo_computo = cfg.asr.tipo_computo

    if dispositivo == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                log.warning("CUDA no disponible: se usa CPU (será mucho más lento)")
                dispositivo = "cpu"
        except Exception:
            dispositivo = "cpu"

    if dispositivo == "cpu" and tipo_computo not in ("int8", "float32"):
        # int8_float16 no existe en CPU.
        tipo_computo = "int8"

    log.info(
        "Cargando WhisperX %s en %s (compute_type=%s)",
        cfg.asr.modelo,
        dispositivo,
        tipo_computo,
    )
    return whisperx.load_model(
        cfg.asr.modelo,
        device=dispositivo,
        compute_type=tipo_computo,
        language=idioma,
        vad_method="silero" if cfg.asr.usar_vad else None,
    )


def transcribir(
    cfg: Config,
    ruta_wav: Path,
    *,
    idioma_cli: str | None = None,
    ruta_original: Path | None = None,
) -> ResultadoAsr:
    """Transcribe (y alinea si procede) el WAV normalizado."""
    import whisperx

    audio = whisperx.load_audio(str(ruta_wav))

    forzado = idioma_forzado(ruta_original or ruta_wav, idioma_cli)

    modelo = cargar_modelo(cfg, idioma=None)
    dispositivo = getattr(modelo, "device", cfg.asr.dispositivo)

    try:
        if forzado:
            idioma_detectado, confianza = None, None
            idioma_usado = forzado
            log.info("Idioma forzado: %s (no se detecta)", idioma_usado)
        else:
            idioma_detectado, confianza = detectar_idioma(modelo, audio, cfg)
            idioma_usado = decidir_idioma(cfg, idioma_detectado, confianza)
            log.info(
                "Idioma detectado=%s (%.2f) -> se usa %s",
                idioma_detectado,
                confianza if confianza is not None else float("nan"),
                idioma_usado,
            )

        log.info("Transcribiendo (batch_size=%d)...", cfg.asr.tamano_lote)
        resultado = modelo.transcribe(
            audio, batch_size=cfg.asr.tamano_lote, language=idioma_usado
        )
    finally:
        liberar_memoria(modelo)
        modelo = None

    segmentos_brutos = resultado.get("segments", []) or []
    hubo_alineamiento = False

    debe_alinear = (
        cfg.idioma.alinear
        and idioma_usado not in cfg.idioma.idiomas_sin_alineamiento
        and bool(segmentos_brutos)
    )

    if debe_alinear:
        try:
            log.info("Alineando a nivel de palabra (%s)...", idioma_usado)
            modelo_align, metadatos = whisperx.load_align_model(
                language_code=idioma_usado, device=dispositivo
            )
            try:
                resultado = whisperx.align(
                    segmentos_brutos,
                    modelo_align,
                    metadatos,
                    audio,
                    dispositivo,
                    return_char_alignments=False,
                )
                segmentos_brutos = resultado.get("segments", []) or segmentos_brutos
                hubo_alineamiento = True
            finally:
                liberar_memoria(modelo_align)
        except Exception as exc:
            log.warning(
                "El alineamiento falló (%s): se sigue con tiempos de segmento", exc
            )
            hubo_alineamiento = False
    elif idioma_usado in cfg.idioma.idiomas_sin_alineamiento:
        log.info(
            "Sin modelo de alineamiento para '%s': se asignará hablante por segmento",
            idioma_usado,
        )

    segmentos = [
        Segmento(
            inicio=float(s.get("start", 0.0) or 0.0),
            fin=float(s.get("end", 0.0) or 0.0),
            texto=(s.get("text") or "").strip(),
            palabras=list(s.get("words") or []),
        )
        for s in segmentos_brutos
    ]

    return ResultadoAsr(
        segmentos=segmentos,
        idioma_detectado=idioma_detectado,
        confianza_idioma=confianza,
        idioma_usado=idioma_usado,
        hubo_alineamiento=hubo_alineamiento,
    )
