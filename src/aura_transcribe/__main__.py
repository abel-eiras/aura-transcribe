"""CLI del pipeline de transcripción de Aura.

Ejemplos:
    aura-transcribe run --dry-run
    aura-transcribe run
    aura-transcribe run --file aura_20260906_161901.wav --keep-audio --no-diarization
    aura-transcribe run --language gl --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import VERSION_PIPELINE
from .config import (
    ErrorConfig,
    cargar_config,
    cargar_token_hf,
    exigir_token_openrouter,
)
from .filesystem import ErrorLock, ErrorMontaje
from .logging_setup import configurar_logging
from .notas import OpcionesNotas, ejecutar_notas
from .pipeline import Opciones, ejecutar_con_lock

CODIGO_OK = 0
CODIGO_ERROR = 1
CODIGO_BLOQUEADO = 2
CODIGO_CONFIG = 3


def construir_parser() -> argparse.ArgumentParser:
    # Opciones comunes, admitidas antes y después del subcomando.
    comunes = argparse.ArgumentParser(add_help=False)
    comunes.add_argument("-v", "--verbose", action="store_true", help="Log más detallado")
    comunes.add_argument(
        "-q", "--quiet", action="store_true", help="Sin salida por consola (solo al log)"
    )

    parser = argparse.ArgumentParser(
        parents=[comunes],
        prog="aura-transcribe",
        description="Transcribe con diarización los audios de Aura, en local.",
    )
    parser.add_argument(
        "--version", action="version", version=f"aura-transcribe {VERSION_PIPELINE}"
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Ruta alternativa a config.toml"
    )
    subcomandos = parser.add_subparsers(dest="comando")

    run = subcomandos.add_parser(
        "run", parents=[comunes], help="Procesa los audios pendientes"
    )
    run.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Procesa solo este audio (ruta absoluta o relativa a la carpeta de trabajo)",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="Rehace la transcripción aunque ya exista el JSON",
    )
    run.add_argument(
        "--no-diarization",
        action="store_true",
        help="Salta la diarización (no hace falta token de Hugging Face)",
    )
    run.add_argument(
        "--language",
        default=None,
        help="Fuerza el idioma (p. ej. es, gl) en vez de detectarlo",
    )
    run.add_argument(
        "--keep-audio",
        action="store_true",
        help="No mueve el audio a processed/ al terminar",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo lista lo que se procesaría; no escribe ni mueve nada",
    )
    run.add_argument(
        "--limit", type=int, default=None, help="Procesa como mucho N audios"
    )
    run.add_argument(
        "--sin-aislamiento",
        action="store_true",
        help=(
            "Procesa en este mismo proceso en vez de en un subproceso por audio "
            "(solo para depurar: es lo que provoca los OOM de VRAM en lotes largos)"
        ),
    )

    subcomandos.add_parser(
        "check", parents=[comunes], help="Comprueba entorno, GPU, token y rutas"
    )

    notas = subcomandos.add_parser(
        "notas",
        parents=[comunes],
        help="Genera notas de Obsidian (acta) a partir de las transcripciones",
    )
    notas.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Procesa solo este JSON (ruta absoluta o relativa a transcription/)",
    )
    notas.add_argument(
        "--backlog",
        action="store_true",
        help="Incluye también transcription/sent/ (histórico), no solo lo nuevo",
    )
    notas.add_argument(
        "--force",
        action="store_true",
        help="Regenera la nota aunque ya exista",
    )

    return parser


def _comando_check(cfg) -> int:
    """Diagnóstico rápido del entorno, sin tocar ningún fichero."""
    log = logging.getLogger("aura.check")
    todo_ok = True

    log.info("Carpeta de trabajo: %s (%s)", cfg.rutas.base, "existe" if cfg.rutas.base.is_dir() else "NO EXISTE")
    if not cfg.rutas.base.is_dir():
        todo_ok = False

    if cfg.rutas.punto_montaje is not None:
        from .filesystem import esta_montado

        montado = esta_montado(cfg.rutas.punto_montaje)
        log.info(
            "Montaje %s: %s", cfg.rutas.punto_montaje, "montado" if montado else "NO MONTADO"
        )
        todo_ok = todo_ok and montado
    else:
        log.info("Punto de montaje: no configurado (sin comprobación)")

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            log.info(
                "GPU: %s, %.1f GB, capacidad %d.%d",
                props.name,
                props.total_memory / 1e9,
                props.major,
                props.minor,
            )
        else:
            log.warning("CUDA no disponible: el pipeline irá por CPU (muy lento)")
    except Exception as exc:
        log.error("torch no está disponible: %s", exc)
        todo_ok = False

    import shutil as _shutil

    for binario in ("ffmpeg", "ffprobe"):
        ruta = _shutil.which(binario)
        log.info("%s: %s", binario, ruta or "NO ENCONTRADO")
        todo_ok = todo_ok and bool(ruta)

    if cargar_token_hf():
        log.info("Token de Hugging Face: encontrado (no se muestra)")
    else:
        log.warning(
            "Token de Hugging Face: NO encontrado; la diarización no funcionará "
            "(usa --no-diarization mientras tanto)"
        )

    log.info("Modelo ASR: %s (%s)", cfg.asr.modelo, cfg.asr.tipo_computo)
    log.info("Modelo de diarización: %s", cfg.diarizacion.modelo)
    return CODIGO_OK if todo_ok else CODIGO_ERROR


def _comando_notas(cfg, args) -> int:
    log = logging.getLogger("aura.notas")

    if not cfg.notas.activo:
        log.error(
            "[notas] activo = false en config.toml: no se genera nada. "
            "Actívalo a propósito (implica mandar las transcripciones a "
            "OpenRouter)."
        )
        return CODIGO_CONFIG

    try:
        api_key = exigir_token_openrouter()
    except ErrorConfig as exc:
        log.error("%s", exc)
        return CODIGO_CONFIG

    opciones = OpcionesNotas(fichero=args.file, backlog=args.backlog, forzar=args.force)
    resultado = ejecutar_notas(cfg, opciones, api_key)

    log.info(
        "Resumen: %d notas generadas, %d omitidas, %d fallidas",
        len(resultado.generadas),
        len(resultado.omitidas),
        len(resultado.fallidas),
    )
    return CODIGO_ERROR if resultado.fallidas else CODIGO_OK


def main(argv: list[str] | None = None) -> int:
    parser = construir_parser()
    args = parser.parse_args(argv)

    if args.comando is None:
        parser.print_help()
        return CODIGO_OK

    try:
        cfg = cargar_config(args.config)
    except ErrorConfig as exc:
        print(f"Configuración inválida: {exc}", file=sys.stderr)
        return CODIGO_CONFIG

    log = configurar_logging(
        cfg.rutas.fichero_log, verboso=args.verbose, silencioso=args.quiet
    )

    if args.comando == "check":
        return _comando_check(cfg)

    if args.comando == "notas":
        return _comando_notas(cfg, args)

    opciones = Opciones(
        fichero=args.file,
        forzar=args.force,
        sin_diarizacion=args.no_diarization,
        idioma=args.language,
        conservar_audio=args.keep_audio,
        dry_run=args.dry_run,
        limite=args.limit,
        ruta_config=args.config,
        sin_aislamiento=args.sin_aislamiento,
        verboso=args.verbose,
    )

    # Aviso temprano y claro si falta el token y se va a diarizar.
    if not opciones.sin_diarizacion and cfg.diarizacion.activa and not opciones.dry_run:
        if not cargar_token_hf():
            from .config import MENSAJE_SIN_TOKEN

            log.error("%s", MENSAJE_SIN_TOKEN)
            return CODIGO_CONFIG

    try:
        resultado = ejecutar_con_lock(cfg, opciones)
    except ErrorLock:
        return CODIGO_BLOQUEADO
    except ErrorMontaje as exc:
        log.error("%s", exc)
        return CODIGO_ERROR
    except ErrorConfig as exc:
        log.error("%s", exc)
        return CODIGO_CONFIG
    except KeyboardInterrupt:
        log.warning("Cancelado.")
        return CODIGO_ERROR
    except Exception as exc:
        log.exception("Fallo inesperado: %s", exc)
        return CODIGO_ERROR

    if not opciones.dry_run:
        log.info(
            "Resumen: %d procesados, %d fallidos, %d ignorados",
            len(resultado.procesados),
            len(resultado.fallidos),
            len(resultado.omitidos),
        )
    return CODIGO_ERROR if resultado.hubo_fallos else CODIGO_OK


if __name__ == "__main__":
    raise SystemExit(main())
