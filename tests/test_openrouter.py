"""Tests del cliente mínimo de OpenRouter (sin red real)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from aura_transcribe.openrouter import ErrorOpenRouter, completar_chat


class _RespuestaFalsa:
    def __init__(self, payload: dict):
        self._cuerpo = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._cuerpo

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_completar_chat_devuelve_el_texto(monkeypatch):
    capturada = {}

    def _urlopen_falso(peticion, timeout=None):
        capturada["peticion"] = peticion
        return _RespuestaFalsa({"choices": [{"message": {"content": "hola"}}]})

    monkeypatch.setattr(
        "aura_transcribe.openrouter.urllib.request.urlopen", _urlopen_falso
    )

    resultado = completar_chat(
        api_key="clave-secreta",
        modelo="modelo-x",
        mensajes=[{"role": "user", "content": "hola"}],
    )

    assert resultado == "hola"
    assert capturada["peticion"].get_header("Authorization") == "Bearer clave-secreta"


def test_respuesta_sin_formato_esperado_lanza_error(monkeypatch):
    def _urlopen_falso(peticion, timeout=None):
        return _RespuestaFalsa({"algo": "raro"})

    monkeypatch.setattr(
        "aura_transcribe.openrouter.urllib.request.urlopen", _urlopen_falso
    )

    with pytest.raises(ErrorOpenRouter):
        completar_chat(api_key="clave", modelo="modelo-x", mensajes=[])


def test_error_http_se_traduce_a_error_openrouter(monkeypatch):
    def _urlopen_falso(peticion, timeout=None):
        raise urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=429,
            msg="rate limit",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b"demasiadas peticiones"),
        )

    monkeypatch.setattr(
        "aura_transcribe.openrouter.urllib.request.urlopen", _urlopen_falso
    )

    with pytest.raises(ErrorOpenRouter, match="429"):
        completar_chat(api_key="clave", modelo="modelo-x", mensajes=[])


def test_error_de_red_se_traduce_a_error_openrouter(monkeypatch):
    def _urlopen_falso(peticion, timeout=None):
        raise urllib.error.URLError("sin conexión")

    monkeypatch.setattr(
        "aura_transcribe.openrouter.urllib.request.urlopen", _urlopen_falso
    )

    with pytest.raises(ErrorOpenRouter):
        completar_chat(api_key="clave", modelo="modelo-x", mensajes=[])
