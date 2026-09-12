"""Tests de la orquestación, sin tocar la GPU ni modelos reales.

Se sustituye la etapa de ASR por una versión de mentira para poder comprobar
el orden de las operaciones, la escritura atómica, el movimiento del audio y
el manejo de fallos con reintentos.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from aura_transcribe import asr as modulo_asr
from aura_transcribe import audio as modulo_audio
from aura_transcribe import pipeline as modulo_pipeline
from aura_transcribe.config import Config
from aura_transcribe.merge import Turno
from aura_transcribe.output import Segmento
from aura_transcribe.pipeline import Opciones, ejecutar


@pytest.fixture()
def sin_modelos(monkeypatch):
    """Sustituye ffprobe/ffmpeg/ASR/diarización por dobles deterministas."""

    def falso_duracion(ruta):
        return 24.08

    def falso_normalizar(origen, destino, frecuencia=16000):
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        Path(destino).write_bytes(b"RIFFfalso")
        return Path(destino)

    def falso_transcribir(cfg, ruta_wav, *, idioma_cli=None, ruta_original=None):
        return modulo_asr.ResultadoAsr(
            segmentos=[
                Segmento(inicio=0.0, fin=2.0, texto="Hola qué tal."),
                Segmento(inicio=2.5, fin=4.0, texto="Muy bien, gracias."),
            ],
            idioma_detectado="es",
            confianza_idioma=0.93,
            idioma_usado=idioma_cli or "es",
            hubo_alineamiento=True,
        )

    def falso_diarizar(cfg, ruta_wav):
        return [Turno(0.0, 2.2, "SPEAKER_01"), Turno(2.2, 4.0, "SPEAKER_00")]

    monkeypatch.setattr(modulo_pipeline, "duracion_segundos", falso_duracion)
    monkeypatch.setattr(modulo_pipeline, "normalizar", falso_normalizar)
    monkeypatch.setattr(modulo_asr, "transcribir", falso_transcribir)
    monkeypatch.setattr(
        "aura_transcribe.diarize.diarizar", falso_diarizar, raising=False
    )
    return falso_transcribir


def _audio(cfg: Config, nombre: str = "aura_prueba.wav") -> Path:
    ruta = cfg.rutas.base / nombre
    ruta.write_bytes(b"x" * 2048)
    viejo = time.time() - 3600
    os.utime(ruta, (viejo, viejo))
    return ruta


def test_flujo_completo(cfg: Config, sin_modelos):
    audio = _audio(cfg)
    resultado = ejecutar(cfg, Opciones())

    assert len(resultado.procesados) == 1
    assert not resultado.fallidos

    destino_json = cfg.rutas.transcripciones / "aura_prueba.json"
    assert destino_json.is_file()

    doc = json.loads(destino_json.read_text(encoding="utf-8"))
    assert doc["metadata"]["source_file"] == "aura_prueba.wav"
    assert doc["metadata"]["num_speakers_detected"] == 2
    assert doc["metadata"]["alignment"] is True
    # Renumerado por orden de aparición: el primero siempre es SPEAKER_00.
    assert doc["segments"][0]["speaker"] == "SPEAKER_00"

    # El audio se movió solo después de escribir el JSON.
    assert not audio.exists()
    assert (cfg.rutas.procesados / "aura_prueba.wav").is_file()


def test_keep_audio_no_mueve(cfg: Config, sin_modelos):
    audio = _audio(cfg)
    ejecutar(cfg, Opciones(conservar_audio=True))
    assert audio.exists()
    assert not (cfg.rutas.procesados / "aura_prueba.wav").exists()


def test_sin_diarizacion_todo_a_speaker_00(cfg: Config, sin_modelos):
    _audio(cfg)
    ejecutar(cfg, Opciones(sin_diarizacion=True))
    doc = json.loads(
        (cfg.rutas.transcripciones / "aura_prueba.json").read_text(encoding="utf-8")
    )
    assert {s["speaker"] for s in doc["segments"]} == {"SPEAKER_00"}
    assert doc["metadata"]["diarization_model"] is None


def test_language_llega_al_json(cfg: Config, sin_modelos):
    _audio(cfg)
    ejecutar(cfg, Opciones(idioma="gl"))
    doc = json.loads(
        (cfg.rutas.transcripciones / "aura_prueba.json").read_text(encoding="utf-8")
    )
    assert doc["metadata"]["language_used"] == "gl"


def test_dry_run_no_escribe_ni_mueve(cfg: Config, sin_modelos):
    audio = _audio(cfg)
    ejecutar(cfg, Opciones(dry_run=True))
    assert audio.exists()
    assert list(cfg.rutas.transcripciones.glob("*.json")) == []


def test_limite(cfg: Config, sin_modelos):
    _audio(cfg, "a.wav")
    _audio(cfg, "b.wav")
    resultado = ejecutar(cfg, Opciones(limite=1))
    assert len(resultado.procesados) == 1


def test_no_reprocesa_lo_ya_hecho(cfg: Config, sin_modelos):
    _audio(cfg)
    ejecutar(cfg, Opciones())
    resultado = ejecutar(cfg, Opciones())
    assert resultado.procesados == []


def test_fallo_deja_el_audio_en_su_sitio_y_cuenta_reintentos(
    cfg: Config, sin_modelos, monkeypatch
):
    def revienta(*args, **kwargs):
        raise RuntimeError("boom de prueba")

    monkeypatch.setattr(modulo_asr, "transcribir", revienta)
    audio = _audio(cfg)

    resultado = ejecutar(cfg, Opciones())

    assert len(resultado.fallidos) == 1
    assert audio.exists(), "el audio nunca se pierde"
    assert not (cfg.rutas.transcripciones / "aura_prueba.json").exists()

    estado = json.loads(cfg.rutas.fichero_estado.read_text(encoding="utf-8"))
    assert estado["aura_prueba.wav"]["intentos"] == 1


def test_tras_agotar_los_reintentos_va_a_failed(cfg: Config, sin_modelos, monkeypatch):
    def revienta(*args, **kwargs):
        raise RuntimeError("boom de prueba")

    monkeypatch.setattr(modulo_asr, "transcribir", revienta)
    cfg = replace(cfg, errores=replace(cfg.errores, intentos_maximos=2))
    audio = _audio(cfg)

    ejecutar(cfg, Opciones())
    assert audio.exists()
    ejecutar(cfg, Opciones())

    assert not audio.exists()
    movido = cfg.rutas.fallidos / "aura_prueba.wav"
    assert movido.is_file(), "el audio se conserva en failed/, nunca se borra"
    error = cfg.rutas.fallidos / "aura_prueba.wav.error.txt"
    assert "boom de prueba" in error.read_text(encoding="utf-8")


def test_montaje_ausente_aborta(cfg: Config, sin_modelos):
    from aura_transcribe.filesystem import ErrorMontaje

    cfg = replace(
        cfg, rutas=replace(cfg.rutas, punto_montaje=cfg.rutas.base / "no-montado")
    )
    _audio(cfg)
    with pytest.raises(ErrorMontaje):
        ejecutar(cfg, Opciones())


def test_file_concreto(cfg: Config, sin_modelos):
    _audio(cfg, "uno.wav")
    _audio(cfg, "dos.wav")
    ejecutar(cfg, Opciones(fichero=Path("uno.wav")))
    assert (cfg.rutas.transcripciones / "uno.json").exists()
    assert not (cfg.rutas.transcripciones / "dos.json").exists()


def test_force_rehace(cfg: Config, sin_modelos):
    _audio(cfg, "uno.wav")
    ejecutar(cfg, Opciones())
    # El audio ya está en processed/; se rehace desde allí con --file --force.
    resultado = ejecutar(
        cfg,
        Opciones(
            fichero=cfg.rutas.procesados / "uno.wav", forzar=True, conservar_audio=True
        ),
    )
    assert len(resultado.procesados) == 1
