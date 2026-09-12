"""Clasificación de una transcripción en uno de los tipos de nota configurados.

Los tipos de nota (`cfg.notas.categorias`) son enteramente configurables (ver
config.example.toml); aquí no hay ninguna categoría hardcodeada. El LLM ve el
`full_text` completo y el `criterio` de cada categoría, y decide con eso: la
señal de idioma (`metadata.language_used`) se le da como pista, nunca como
regla, para no fallar en un caso raro (p. ej. una nota de trabajo en el idioma
"equivocado").
"""

from __future__ import annotations

from .config import CategoriaNota, Config
from .openrouter import completar_chat


def _prompt_sistema(categorias: tuple[CategoriaNota, ...]) -> str:
    ids = ", ".join(c.id for c in categorias)
    lineas_criterio = "\n".join(f"- {c.id}: {c.criterio}." for c in categorias)
    return (
        f"Clasificas transcripciones de notas de voz en uno de estos tipos "
        f"EXACTOS:\n\n{lineas_criterio}\n\n"
        f"Responde ÚNICAMENTE con uno de estos identificadores, en minúsculas "
        f"y sin nada más: {ids}."
    )


class ErrorClasificacion(RuntimeError):
    """La respuesta del LLM no fue uno de los identificadores configurados."""


def clasificar(cfg: Config, documento: dict, api_key: str) -> str:
    """Devuelve el id de categoría de `documento` (metadata + segments + full_text)."""
    categorias = cfg.notas.categorias
    idioma = documento.get("metadata", {}).get("language_used", "?")
    texto = documento.get("full_text", "")

    respuesta = completar_chat(
        api_key=api_key,
        modelo=cfg.notas.modelo,
        mensajes=[
            {"role": "system", "content": _prompt_sistema(categorias)},
            {"role": "user", "content": f"Idioma detectado: {idioma}\n\n{texto}"},
        ],
        temperatura=0.0,
    )

    categoria = respuesta.strip().lower()
    ids_validos = {c.id for c in categorias}
    if categoria not in ids_validos:
        raise ErrorClasificacion(
            f"Respuesta de clasificación inesperada: {respuesta!r}"
        )
    return categoria
