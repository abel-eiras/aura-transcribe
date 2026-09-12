"""Cliente mínimo de la API de OpenRouter (chat completions).

Sin dependencias nuevas: usa `urllib` de la librería estándar, igual de bien
para el volumen de peticiones de este proyecto (unas pocas al día como mucho).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

URL_CHAT = "https://openrouter.ai/api/v1/chat/completions"


class ErrorOpenRouter(RuntimeError):
    """La llamada a OpenRouter falló: red, rate limit, o respuesta sin el
    formato esperado. Se trata como un fallo normal del audio en curso: no se
    escribe nada a medias y se reintenta en la siguiente pasada."""


def completar_chat(
    *,
    api_key: str,
    modelo: str,
    mensajes: list[dict[str, str]],
    temperatura: float = 0.2,
    timeout: float = 120.0,
) -> str:
    """Llama al endpoint de chat completions y devuelve el texto de la respuesta."""
    cuerpo = json.dumps(
        {"model": modelo, "messages": mensajes, "temperature": temperatura}
    ).encode("utf-8")
    peticion = urllib.request.Request(
        URL_CHAT,
        data=cuerpo,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "aura-transcribe",
        },
    )
    try:
        with urllib.request.urlopen(peticion, timeout=timeout) as resp:  # noqa: S310
            datos = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detalle = exc.read().decode("utf-8", errors="replace")
        raise ErrorOpenRouter(f"OpenRouter devolvió {exc.code}: {detalle[:500]}") from exc
    except urllib.error.URLError as exc:
        raise ErrorOpenRouter(f"No se pudo contactar con OpenRouter: {exc}") from exc

    try:
        return datos["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ErrorOpenRouter(
            f"Respuesta de OpenRouter sin el formato esperado: {datos}"
        ) from exc
