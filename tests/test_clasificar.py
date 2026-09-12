"""Tests de clasificación de transcripciones (API de OpenRouter mockeada)."""

from __future__ import annotations

import pytest

from aura_transcribe.clasificar import ErrorClasificacion, clasificar


def _documento(idioma: str = "es", texto: str = "hola") -> dict:
    return {
        "metadata": {"language_used": idioma},
        "segments": [],
        "full_text": texto,
    }


def test_clasifica_segun_la_respuesta_del_modelo(cfg, monkeypatch):
    monkeypatch.setattr(
        "aura_transcribe.clasificar.completar_chat", lambda **_: "visita_cliente"
    )
    assert clasificar(cfg, _documento(), "clave") == "visita_cliente"


def test_respuesta_con_espacios_o_mayusculas_se_normaliza(cfg, monkeypatch):
    monkeypatch.setattr(
        "aura_transcribe.clasificar.completar_chat", lambda **_: "  Reunion \n"
    )
    assert clasificar(cfg, _documento(), "clave") == "reunion"


def test_respuesta_invalida_lanza_error(cfg, monkeypatch):
    monkeypatch.setattr(
        "aura_transcribe.clasificar.completar_chat", lambda **_: "no lo sé"
    )
    with pytest.raises(ErrorClasificacion):
        clasificar(cfg, _documento(), "clave")


def test_usa_el_modelo_de_la_config(cfg, monkeypatch):
    capturado = {}

    def _completar_falso(**kwargs):
        capturado["modelo"] = kwargs["modelo"]
        return "reunion"

    monkeypatch.setattr("aura_transcribe.clasificar.completar_chat", _completar_falso)
    clasificar(cfg, _documento(), "clave")
    assert capturado["modelo"] == cfg.notas.modelo


def test_el_prompt_incluye_todas_las_categorias_configuradas(cfg, monkeypatch):
    capturado = {}

    def _completar_falso(**kwargs):
        capturado["mensajes"] = kwargs["mensajes"]
        return "reunion"

    monkeypatch.setattr("aura_transcribe.clasificar.completar_chat", _completar_falso)
    clasificar(cfg, _documento(), "clave")
    prompt_sistema = capturado["mensajes"][0]["content"]
    for categoria in cfg.notas.categorias:
        assert categoria.id in prompt_sistema
