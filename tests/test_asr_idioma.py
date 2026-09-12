"""Tests de la regla de idioma restringido (es/gl en la fixture `cfg`, pero el
mecanismo es genérico: ver `[idioma]` en config.example.toml)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from aura_transcribe.asr import decidir_idioma, idioma_forzado


def test_espanol_normal(cfg):
    assert decidir_idioma(cfg, "es", 0.95) == "es"


def test_alternativo_por_encima_del_umbral(cfg):
    assert decidir_idioma(cfg, "gl", 0.7) == "gl"


def test_alternativo_por_debajo_del_umbral_cae_al_por_defecto(cfg):
    assert decidir_idioma(cfg, "gl", 0.2) == "es"


def test_confundible_con_confianza_alta_se_trata_como_el_alternativo(cfg):
    assert decidir_idioma(cfg, "pt", 0.8) == "gl"


def test_deteccion_nunca_es_libre(cfg):
    # Un audio corto y ruidoso detectado como italiano no debe transcribirse en italiano.
    assert decidir_idioma(cfg, "it", 0.99) == "es"
    assert decidir_idioma(cfg, "en", 0.99) == "es"


def test_sin_deteccion_cae_al_idioma_por_defecto(cfg):
    assert decidir_idioma(cfg, None, None) == "es"
    assert decidir_idioma(cfg, "gl", None) == "es"


def test_umbral_configurable(cfg):
    estricto = replace(cfg, idioma=replace(cfg.idioma, umbral_alternativo=0.9))
    assert decidir_idioma(estricto, "gl", 0.7) == "es"


def test_alternativo_no_permitido_cae_al_por_defecto(cfg):
    solo_es = replace(cfg, idioma=replace(cfg.idioma, permitidos=("es",)))
    assert decidir_idioma(solo_es, "gl", 0.99) == "es"


def test_sin_idioma_alternativo_configurado_todo_cae_al_por_defecto(cfg):
    sin_alternativo = replace(cfg, idioma=replace(cfg.idioma, idioma_alternativo=""))
    assert decidir_idioma(sin_alternativo, "gl", 0.99) == "es"


def test_override_por_cli_manda(tmp_path: Path):
    audio = tmp_path / "nota_gl.wav"
    audio.write_bytes(b"x")
    assert idioma_forzado(audio, "es") == "es"


def test_override_por_sufijo_del_nombre(tmp_path: Path):
    audio = tmp_path / "nota_20260906_161901_gl.wav"
    audio.write_bytes(b"x")
    assert idioma_forzado(audio, None) == "gl"


def test_override_por_sidecar(tmp_path: Path):
    audio = tmp_path / "nota_x.wav"
    audio.write_bytes(b"x")
    (tmp_path / "nota_x.lang").write_text("gl\n", encoding="utf-8")
    assert idioma_forzado(audio, None) == "gl"


def test_el_sidecar_manda_sobre_el_sufijo(tmp_path: Path):
    audio = tmp_path / "nota_x_gl.wav"
    audio.write_bytes(b"x")
    (tmp_path / "nota_x_gl.lang").write_text("es", encoding="utf-8")
    assert idioma_forzado(audio, None) == "es"


def test_sin_override(tmp_path: Path):
    audio = tmp_path / "nota_20260906_161901.wav"
    audio.write_bytes(b"x")
    assert idioma_forzado(audio, None) is None
