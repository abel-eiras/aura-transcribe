"""Configuración del logging: consola + fichero rotado."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

FORMATO = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"


def configurar_logging(
    fichero_log: Path | None = None,
    *,
    verboso: bool = False,
    silencioso: bool = False,
    rotar: bool = True,
) -> logging.Logger:
    """Deja el logging raíz listo y devuelve el logger del pipeline.

    `rotar=False` lo usan los subprocesos de trabajo: padre e hijo escriben en el
    mismo fichero a la vez y solo uno debe encargarse de rotarlo. El hijo abre en
    modo append (write() de líneas cortas con O_APPEND es atómico en Linux).
    """
    nivel = logging.DEBUG if verboso else logging.INFO
    raiz = logging.getLogger()
    raiz.setLevel(logging.DEBUG)

    for manejador in list(raiz.handlers):
        raiz.removeHandler(manejador)

    formateador = logging.Formatter(FORMATO, datefmt=FORMATO_FECHA)

    if not silencioso:
        consola = logging.StreamHandler()
        consola.setLevel(nivel)
        consola.setFormatter(formateador)
        raiz.addHandler(consola)

    if fichero_log is not None:
        fichero_log = Path(fichero_log)
        try:
            fichero_log.parent.mkdir(parents=True, exist_ok=True)
            a_fichero = (
                logging.handlers.RotatingFileHandler(
                    fichero_log,
                    maxBytes=5 * 1024 * 1024,
                    backupCount=5,
                    encoding="utf-8",
                )
                if rotar
                else logging.FileHandler(fichero_log, encoding="utf-8")
            )
            a_fichero.setLevel(logging.DEBUG)
            a_fichero.setFormatter(formateador)
            raiz.addHandler(a_fichero)
        except OSError as exc:  # p. ej. disco lleno: no debe tumbar el pipeline
            logging.getLogger("aura").warning(
                "No se pudo abrir el log %s: %s", fichero_log, exc
            )

    # Las librerías de terceros son muy habladoras; se bajan a WARNING.
    for ruidoso in (
        "urllib3",
        "huggingface_hub",
        "filelock",
        "speechbrain",
        "pyannote",
        "pytorch_lightning",
        "matplotlib",
        "numba",
        "asyncio",
        "torio",
        "torchaudio",
        "torch",
        "fsspec",
        "lightning_fabric",
    ):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    return logging.getLogger("aura")
