"""Construcción y escritura del JSON de salida.

Se respeta el esquema que ya consumen los JSON de transcription/sent/
(metadata + segments + full_text) y solo se AÑADEN campos nuevos, todos
compatibles hacia atrás (sección 6 del plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import VERSION_PIPELINE
from .config import Config
from .filesystem import escribir_json_atomico, escribir_texto_atomico

ETIQUETA_SIN_HABLANTE = "SPEAKER_00"


@dataclass
class Segmento:
    """Un segmento de transcripción ya fusionado con su hablante."""

    inicio: float
    fin: float
    texto: str
    hablante: str | None = None
    palabras: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duracion(self) -> float:
        return max(0.0, self.fin - self.inicio)


def _redondear(valor: float, decimales: int = 3) -> float:
    return round(float(valor), decimales)


def construir_documento(
    *,
    cfg: Config,
    nombre_origen: str,
    duracion_segundos: float,
    idioma_detectado: str | None,
    confianza_idioma: float | None,
    idioma_usado: str,
    segmentos: list[Segmento],
    hubo_alineamiento: bool,
    diarizacion_activa: bool,
    segundos_proceso: float,
    procesado_en: datetime | None = None,
) -> dict[str, Any]:
    """Devuelve el diccionario listo para serializar."""
    momento = procesado_en or datetime.now(timezone.utc)

    hablantes = {s.hablante for s in segmentos if s.hablante}
    num_hablantes = len(hablantes)

    metadata: dict[str, Any] = {
        "source_file": nombre_origen,
        "processed_at": momento.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": _redondear(duracion_segundos, 2),
        "language_detected": idioma_detectado,
        "language_confidence": (
            _redondear(confianza_idioma, 4) if confianza_idioma is not None else None
        ),
        "language_used": idioma_usado,
        "whisper_model": cfg.asr.modelo,
        "num_speakers_detected": num_hablantes,
        # Campos nuevos (sección 6 del plan), compatibles hacia atrás.
        "diarization_model": cfg.diarizacion.modelo if diarizacion_activa else None,
        "alignment": hubo_alineamiento,
        "pipeline_version": VERSION_PIPELINE,
        "processing_seconds": _redondear(segundos_proceso, 2),
    }

    lista_segmentos: list[dict[str, Any]] = []
    for indice, seg in enumerate(segmentos):
        entrada: dict[str, Any] = {
            "id": indice,
            "speaker": seg.hablante or ETIQUETA_SIN_HABLANTE,
            "start": _redondear(seg.inicio),
            "end": _redondear(seg.fin),
            "text": seg.texto,
        }
        if cfg.salida.incluir_palabras and seg.palabras:
            entrada["words"] = [
                {
                    "word": p.get("word", ""),
                    "start": _redondear(p["start"]) if p.get("start") is not None else None,
                    "end": _redondear(p["end"]) if p.get("end") is not None else None,
                    "speaker": p.get("speaker"),
                    "score": (
                        _redondear(p["score"], 4) if p.get("score") is not None else None
                    ),
                }
                for p in seg.palabras
            ]
        lista_segmentos.append(entrada)

    full_text = "\n".join(
        f"{entrada['speaker']}: {entrada['text']}" for entrada in lista_segmentos
    )

    return {"metadata": metadata, "segments": lista_segmentos, "full_text": full_text}


def escribir_salida(cfg: Config, documento: dict[str, Any], destino_json: Path) -> Path:
    """Escribe el JSON (y el .txt opcional) de forma atómica."""
    destino_json = Path(destino_json)
    escribir_json_atomico(destino_json, documento)

    if cfg.salida.escribir_txt:
        destino_txt = destino_json.with_suffix(".txt")
        escribir_texto_atomico(destino_txt, documento.get("full_text", "") + "\n")

    return destino_json
