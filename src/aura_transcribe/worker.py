"""Worker de un solo audio: `python -m aura_transcribe.worker`.

Existe por una razón medida, no teórica. En la GTX 1660 (6 GB) la VRAM se
fragmenta a lo largo de un proceso de vida larga: en una pasada por lotes de 5
audios reales, el cuarto fichero reventó con `CUDA failed with error out of
memory` pese a que asr.py ya hace `gc.collect()` + `torch.cuda.empty_cache()`
entre etapas. Ese mismo fichero, relanzado a solas en un proceso nuevo, terminó
perfectamente. La única forma fiable de devolver TODA la VRAM al sistema es que
el proceso muera, así que cada audio se procesa aquí y este proceso se muere.

Protocolo con el padre (ver `pipeline._procesar_en_subproceso`):
  * STDOUT: una única línea `@@AURA-RESULTADO@@ {json}` con el resultado.
  * STDERR: heredado del padre, no se captura; por ahí salen los logs en vivo.
  * Código de salida: 0 si todo fue bien, 1 si hubo cualquier excepción.

Este proceso NO coge el flock del pipeline: ya lo tiene el padre en
`ejecutar_con_lock()` y volver a pedirlo sería un interbloqueo consigo mismo.
Aquí se llama directamente a `pipeline.procesar_audio()`, que no toca ese lock ni
el fichero de estado.

Sí coge, en cambio, el lock de sincronización de rclone, porque el que muta la
carpeta de trabajo es este proceso: lo hace dentro de `procesar_audio()` y solo
durante la ventana de escritura del JSON + movimiento del audio.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .config import ErrorConfig, cargar_config
from .discovery import resolver_audio_suelto
from .logging_setup import configurar_logging
from .pipeline import (
    CENTINELA_RESULTADO,
    opciones_desde_dict,
    procesar_audio,
    resultado_a_dict,
)

CODIGO_OK = 0
CODIGO_ERROR = 1


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aura-transcribe-worker",
        description="Procesa un único audio en un proceso aislado (uso interno).",
    )
    parser.add_argument("--config", type=Path, required=True, help="Ruta del config.toml")
    parser.add_argument("--audio", type=Path, required=True, help="Audio a procesar")
    parser.add_argument(
        "--opciones", required=True, help="Opciones de la pasada, serializadas en JSON"
    )
    parser.add_argument("-v", "--verboso", action="store_true", help="Log más detallado")
    return parser


def _emitir(carga: dict) -> None:
    """Escribe la línea centinela con el resultado y vacía el buffer."""
    sys.stdout.write(
        f"{CENTINELA_RESULTADO} {json.dumps(carga, ensure_ascii=False)}\n"
    )
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)

    try:
        cfg = cargar_config(args.config)
    except ErrorConfig as exc:
        _emitir({"ok": False, "error": f"Configuración inválida: {exc}", "traceback": ""})
        return CODIGO_ERROR

    # Mismo logging que la CLI, salvo la rotación: el padre escribe en el mismo
    # fichero y solo uno de los dos debe rotarlo.
    log = configurar_logging(
        cfg.rutas.fichero_log, verboso=args.verboso, rotar=False
    )

    try:
        opciones = opciones_desde_dict(json.loads(args.opciones))
        # La ruta del config manda la que nos pasó el padre por la línea de órdenes.
        opciones.ruta_config = Path(args.config)
        opciones.sin_aislamiento = True  # aquí ya estamos aislados

        pendiente = resolver_audio_suelto(cfg, Path(args.audio))
        resultado = procesar_audio(cfg, pendiente, opciones)
    except BaseException as exc:  # incluye KeyboardInterrupt y SystemExit
        detalle = traceback.format_exc()
        # Solo el titular: el traceback completo viaja en el JSON y el padre ya
        # lo registra por su lado. Repetirlo aquí llenaría el log de duplicados.
        log.error("El worker falló con %s: %s", Path(args.audio).name, exc)
        _emitir(
            {
                "ok": False,
                "audio": str(args.audio),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": detalle,
            }
        )
        return CODIGO_ERROR

    _emitir(resultado_a_dict(resultado))
    return CODIGO_OK


if __name__ == "__main__":
    raise SystemExit(main())
