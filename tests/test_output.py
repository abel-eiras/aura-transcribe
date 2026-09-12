"""Tests del esquema de salida JSON."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from aura_transcribe import VERSION_PIPELINE
from aura_transcribe.config import Config
from aura_transcribe.output import Segmento, construir_documento, escribir_salida


def _documento(cfg: Config, **extra):
    parametros = dict(
        cfg=cfg,
        nombre_origen="aura_20260906_161901.wav",
        duracion_segundos=24.08,
        idioma_detectado="es",
        confianza_idioma=0.93456,
        idioma_usado="es",
        segmentos=[
            Segmento(inicio=2.1274, fin=4.41, texto="Vale.", hablante="SPEAKER_00"),
            Segmento(inicio=5.83, fin=7.49, texto="Dime.", hablante="SPEAKER_01"),
        ],
        hubo_alineamiento=True,
        diarizacion_activa=True,
        segundos_proceso=12.44,
        procesado_en=datetime(2026, 9, 8, 10, 0, 0, tzinfo=timezone.utc),
    )
    parametros.update(extra)
    return construir_documento(**parametros)


def test_esquema_compatible_con_los_json_existentes(cfg: Config):
    doc = _documento(cfg)
    assert set(doc) == {"metadata", "segments", "full_text"}

    meta = doc["metadata"]
    # Campos que ya existían en junio: no pueden faltar ni cambiar de nombre.
    for clave in (
        "source_file",
        "processed_at",
        "duration_seconds",
        "language_detected",
        "language_confidence",
        "language_used",
        "whisper_model",
        "num_speakers_detected",
    ):
        assert clave in meta, clave

    assert meta["source_file"] == "aura_20260906_161901.wav"
    assert meta["processed_at"] == "2026-09-08T10:00:00Z"
    assert meta["duration_seconds"] == 24.08
    assert meta["language_confidence"] == 0.9346
    assert meta["num_speakers_detected"] == 2


def test_campos_nuevos_de_la_seccion_6(cfg: Config):
    meta = _documento(cfg)["metadata"]
    assert meta["diarization_model"] == cfg.diarizacion.modelo
    assert meta["alignment"] is True
    assert meta["pipeline_version"] == VERSION_PIPELINE
    assert meta["processing_seconds"] == 12.44


def test_segmentos_y_full_text(cfg: Config):
    doc = _documento(cfg)
    assert doc["segments"][0] == {
        "id": 0,
        "speaker": "SPEAKER_00",
        "start": 2.127,
        "end": 4.41,
        "text": "Vale.",
    }
    assert [s["id"] for s in doc["segments"]] == [0, 1]
    assert doc["full_text"] == "SPEAKER_00: Vale.\nSPEAKER_01: Dime."


def test_sin_diarizacion_no_pone_modelo(cfg: Config):
    doc = _documento(cfg, diarizacion_activa=False)
    assert doc["metadata"]["diarization_model"] is None


def test_segmento_sin_hablante_cae_en_speaker_00(cfg: Config):
    doc = _documento(
        cfg, segmentos=[Segmento(inicio=0, fin=1, texto="x", hablante=None)]
    )
    assert doc["segments"][0]["speaker"] == "SPEAKER_00"
    assert doc["metadata"]["num_speakers_detected"] == 0


def test_palabras_solo_si_se_piden(cfg: Config):
    palabras = [{"word": "Vale", "start": 2.1, "end": 2.5, "score": 0.91}]
    segmentos = [
        Segmento(inicio=2.1, fin=2.5, texto="Vale", hablante="SPEAKER_00", palabras=palabras)
    ]
    assert "words" not in _documento(cfg, segmentos=segmentos)["segments"][0]

    cfg_con = replace(cfg, salida=replace(cfg.salida, incluir_palabras=True))
    doc = _documento(cfg_con, segmentos=segmentos)
    assert doc["segments"][0]["words"][0]["word"] == "Vale"


def test_escritura_del_json_y_del_txt(cfg: Config, tmp_path: Path):
    doc = _documento(cfg)
    destino = cfg.rutas.transcripciones / "aura_x.json"
    escribir_salida(cfg, doc, destino)
    assert json.loads(destino.read_text(encoding="utf-8"))["metadata"]["source_file"]
    assert not destino.with_suffix(".txt").exists()

    cfg_txt = replace(cfg, salida=replace(cfg.salida, escribir_txt=True))
    escribir_salida(cfg_txt, doc, destino)
    assert "SPEAKER_00: Vale." in destino.with_suffix(".txt").read_text(encoding="utf-8")
