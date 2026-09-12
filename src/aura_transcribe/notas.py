"""Genera notas de Obsidian ("acta") a partir de los JSON de transcripción.

Ver la sección "Notas de Obsidian" del README para el diseño completo. Resumen:
  * Clasifica cada transcripción (clasificar.py) en uno de los tipos de nota
    definidos en cfg.notas.categorias (totalmente configurables).
  * Redacta la nota con el modelo de OpenRouter que toque según la categoría
    (cada categoría puede tener su propio override de modelo).
  * Escribe el markdown de forma atómica en vault/<carpeta de la categoría>/.

A diferencia de pipeline.py, esto NO toca nada de la carpeta de trabajo: solo
LEE los JSON de transcription/ y transcription/sent/, y ESCRIBE en el vault
local. No hace falta ningún lock ni aislamiento en subproceso (no hay GPU).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .clasificar import ErrorClasificacion, clasificar
from .config import CategoriaNota, Config
from .filesystem import escribir_texto_atomico
from .openrouter import ErrorOpenRouter, completar_chat

log = logging.getLogger("aura.notas")

_PATRON_FECHA_NOMBRE = re.compile(r"(\d{4})(\d{2})(\d{2})")


def _por_id(cfg: Config) -> dict[str, CategoriaNota]:
    return {c.id: c for c in cfg.notas.categorias}


@dataclass
class OpcionesNotas:
    fichero: Path | None = None
    backlog: bool = False
    forzar: bool = False


@dataclass
class ResultadoNotas:
    generadas: list[Path] = field(default_factory=list)
    omitidas: list[tuple[Path, str]] = field(default_factory=list)
    fallidas: list[tuple[Path, str]] = field(default_factory=list)


def _fecha_desde(documento: dict, nombre_json: str) -> str:
    """Fecha de la nota: la del nombre del audio si sigue el patrón
    <algo>_YYYYMMDD_..., si no la de `processed_at`."""
    coincidencia = _PATRON_FECHA_NOMBRE.search(nombre_json)
    if coincidencia:
        return "-".join(coincidencia.groups())
    return documento.get("metadata", {}).get("processed_at", "")[:10] or "sin-fecha"


def _frontmatter(
    *, categoria: CategoriaNota, fecha: str, documento: dict, ruta_json: Path
) -> str:
    metadata = documento.get("metadata", {})
    duracion_min = round(metadata.get("duration_seconds", 0) / 60, 1)
    lineas = [
        "---",
        f"tipo: {categoria.id}",
        f"fecha: {fecha}",
        f'audio_origen: "{metadata.get("source_file", "")}"',
        f'json_origen: "{ruta_json.name}"',
        f"idioma: {metadata.get('language_used', '')}",
        f"hablantes: {metadata.get('num_speakers_detected', 0)}",
        f"duracion_min: {duracion_min}",
        f"tags: [aura, {categoria.tag}]",
        "---",
        "",
    ]
    return "\n".join(lineas)


def _modelo_para(cfg: Config, categoria: CategoriaNota) -> str:
    """El override de la categoría gana si está puesto; si no, el general."""
    return categoria.modelo or cfg.notas.modelo


def _redactar_cuerpo(cfg: Config, documento: dict, categoria: CategoriaNota, api_key: str) -> str:
    respuesta = completar_chat(
        api_key=api_key,
        modelo=_modelo_para(cfg, categoria),
        mensajes=[
            {"role": "system", "content": categoria.prompt_redaccion},
            {"role": "user", "content": documento.get("full_text", "")},
        ],
        temperatura=0.3,
    )
    return respuesta.strip()


def _nota_existente(cfg: Config, stem: str) -> Path | None:
    """Busca en las carpetas de todas las categorías una nota ya generada para `stem`.

    No hace falta saber la categoría de antemano (evita reclasificar solo
    para comprobar si ya existe): el nombre de fichero siempre acaba en
    `-<stem>.md` sea cual sea la carpeta.
    """
    for categoria in cfg.notas.categorias:
        coincidencias = list((cfg.notas.vault / categoria.carpeta).glob(f"*-{stem}.md"))
        if coincidencias:
            return coincidencias[0]
    return None


def generar_nota(
    cfg: Config, ruta_json: Path, api_key: str, *, forzar: bool = False
) -> Path | None:
    """Genera (o regenera) la nota de un único JSON. Devuelve None si se omite."""
    ruta_json = Path(ruta_json)
    stem = ruta_json.stem

    existente = _nota_existente(cfg, stem)
    if existente is not None and not forzar:
        log.info("%s ya tiene nota (%s); se omite", ruta_json.name, existente)
        return None

    documento = json.loads(ruta_json.read_text(encoding="utf-8"))

    if not documento.get("full_text", "").strip():
        # Audio vacío o silencioso (p. ej. una grabación accidental de <1s):
        # no hay nada que clasificar ni redactar. Ni se llama al modelo. Si ya
        # había una nota de una pasada anterior (antes de este chequeo, o
        # generada por error), se borra: no debería existir.
        if existente is not None:
            log.warning(
                "%s no tiene texto; se borra la nota anterior %s",
                ruta_json.name,
                existente,
            )
            existente.unlink()
        else:
            log.info("%s no tiene texto; se omite", ruta_json.name)
        return None

    categorias = _por_id(cfg)
    id_categoria = clasificar(cfg, documento, api_key)
    categoria = categorias[id_categoria]
    log.info("%s clasificado como %s", ruta_json.name, categoria.id)

    cuerpo = _redactar_cuerpo(cfg, documento, categoria, api_key)
    fecha = _fecha_desde(documento, ruta_json.name)
    frontmatter = _frontmatter(
        categoria=categoria, fecha=fecha, documento=documento, ruta_json=ruta_json
    )
    titulo = f"# {categoria.etiqueta} — {fecha}\n\n"
    texto = frontmatter + titulo + cuerpo + "\n"

    destino = cfg.notas.vault / categoria.carpeta / f"{fecha}-{stem}.md"

    # La clasificación puede variar entre pasadas (sobre todo en casos límite
    # o con --force); si la categoría cambió, la nota vieja queda en otra
    # carpeta y hay que quitarla para no acabar con dos notas del mismo audio.
    if existente is not None and existente != destino:
        log.warning(
            "%s cambió de categoría (antes en %s); se borra la nota vieja",
            ruta_json.name,
            existente,
        )
        existente.unlink()

    escribir_texto_atomico(destino, texto)
    log.info("Nota escrita en %s", destino)
    return destino


def _json_pendientes(cfg: Config, *, backlog: bool) -> list[Path]:
    """JSON de transcription/ y, si `backlog`, también de transcription/sent/.

    `transcripciones.glob("*.json")` no es recursivo: no baja a `sent/`, que
    es justo la subcarpeta que se añade aparte cuando se pide backlog.
    """
    carpetas = [cfg.rutas.transcripciones]
    if backlog:
        carpetas.append(cfg.rutas.enviados)

    pendientes: list[Path] = []
    for carpeta in carpetas:
        if carpeta.is_dir():
            pendientes.extend(sorted(carpeta.glob("*.json")))
    return pendientes


def ejecutar_notas(cfg: Config, opciones: OpcionesNotas, api_key: str) -> ResultadoNotas:
    """Genera notas para todo lo pendiente (o para --file). No toca ningún JSON
    ni audio; lo único que puede borrar es una nota propia desactualizada (ver
    generar_nota). Un fallo en un JSON no interrumpe el resto."""
    resultado = ResultadoNotas()

    if opciones.fichero is not None:
        pendientes = [Path(opciones.fichero)]
    else:
        pendientes = _json_pendientes(cfg, backlog=opciones.backlog)

    for ruta_json in pendientes:
        try:
            destino = generar_nota(cfg, ruta_json, api_key, forzar=opciones.forzar)
        except (ErrorClasificacion, ErrorOpenRouter, json.JSONDecodeError, OSError) as exc:
            log.error("%s: %s", ruta_json.name, exc)
            resultado.fallidas.append((ruta_json, str(exc)))
            continue

        if destino is None:
            resultado.omitidas.append((ruta_json, "omitido (ya tenía nota o sin texto)"))
        else:
            resultado.generadas.append(destino)

    return resultado
