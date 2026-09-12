"""Tests del descubrimiento de audios pendientes."""

from __future__ import annotations

import os
import time
from pathlib import Path

from aura_transcribe.config import Config
from aura_transcribe.discovery import (
    descubrir_pendientes,
    resolver_audio_suelto,
    ruta_json_salida,
    ya_transcrito,
)


def _crear(ruta: Path, contenido: bytes = b"x" * 64, antiguedad: float = 3600.0) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(contenido)
    viejo = time.time() - antiguedad
    os.utime(ruta, (viejo, viejo))
    return ruta


def test_solo_audios_y_se_ignora_el_jpeg(cfg: Config):
    base = cfg.rutas.base
    for nombre in (
        "aura_20260901_130010.wav",
        "aura_20260902_114152.wav",
        "aura_20260902_114159.wav",
        "aura_20260902_115906.wav",
        "aura_20260906_161901.wav",
    ):
        _crear(base / nombre)
    _crear(base / "IMG-20260905-WA0000.jpeg")

    pendientes, descartados = descubrir_pendientes(cfg)

    assert [p.nombre for p in pendientes] == [
        "aura_20260901_130010.wav",
        "aura_20260902_114152.wav",
        "aura_20260902_114159.wav",
        "aura_20260902_115906.wav",
        "aura_20260906_161901.wav",
    ]
    assert all(not p.nombre.endswith(".jpeg") for p in pendientes)
    assert descartados == []


def test_otras_extensiones_de_audio(cfg: Config):
    for nombre in ("a.ogg", "b.m4a", "c.mp3", "d.opus", "e.txt", "f.jpeg"):
        _crear(cfg.rutas.base / nombre)
    pendientes, _ = descubrir_pendientes(cfg)
    assert sorted(p.nombre for p in pendientes) == ["a.ogg", "b.m4a", "c.mp3", "d.opus"]


def test_se_ignora_lo_ya_transcrito(cfg: Config):
    _crear(cfg.rutas.base / "uno.wav")
    _crear(cfg.rutas.base / "dos.wav")
    (cfg.rutas.transcripciones / "uno.json").write_text("{}", encoding="utf-8")
    (cfg.rutas.enviados / "dos.json").write_text("{}", encoding="utf-8")
    _crear(cfg.rutas.base / "tres.wav")

    pendientes, descartados = descubrir_pendientes(cfg)

    assert [p.nombre for p in pendientes] == ["tres.wav"]
    assert {r.name for r, _ in descartados} == {"uno.wav", "dos.wav"}
    assert ya_transcrito(cfg, cfg.rutas.base / "uno.wav")
    assert ya_transcrito(cfg, cfg.rutas.base / "dos.wav")


def test_force_recupera_los_ya_transcritos(cfg: Config):
    _crear(cfg.rutas.base / "uno.wav")
    (cfg.rutas.transcripciones / "uno.json").write_text("{}", encoding="utf-8")
    pendientes, _ = descubrir_pendientes(cfg, forzar=True)
    assert [p.nombre for p in pendientes] == ["uno.wav"]


def test_se_descartan_ficheros_inestables(cfg: Config):
    from dataclasses import replace

    cfg = replace(cfg, audio=replace(cfg.audio, segundos_estabilidad=60.0))
    _crear(cfg.rutas.base / "reciente.wav", antiguedad=1.0)
    _crear(cfg.rutas.base / "asentado.wav", antiguedad=600.0)

    pendientes, descartados = descubrir_pendientes(cfg)

    assert [p.nombre for p in pendientes] == ["asentado.wav"]
    assert descartados[0][0].name == "reciente.wav"
    assert "inestable" in descartados[0][1]


def test_se_descartan_ficheros_vacios(cfg: Config):
    _crear(cfg.rutas.base / "vacio.wav", contenido=b"")
    pendientes, descartados = descubrir_pendientes(cfg)
    assert pendientes == []
    assert descartados[0][0].name == "vacio.wav"


def test_no_mira_dentro_de_subcarpetas(cfg: Config):
    _crear(cfg.rutas.procesados / "antiguo.wav")
    pendientes, _ = descubrir_pendientes(cfg)
    assert pendientes == []


def test_ruta_json_salida(cfg: Config):
    destino = ruta_json_salida(cfg, cfg.rutas.base / "aura_x.wav")
    assert destino == cfg.rutas.transcripciones / "aura_x.json"


def test_resolver_audio_suelto_relativo(cfg: Config):
    _crear(cfg.rutas.base / "suelto.wav")
    pendiente = resolver_audio_suelto(cfg, Path("suelto.wav"))
    assert pendiente.ruta == cfg.rutas.base / "suelto.wav"
    assert pendiente.tamano > 0
