"""Utilidades compartidas por los tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aura_transcribe.config import (  # noqa: E402
    CategoriaNota,
    Config,
    ConfigAsr,
    ConfigAudio,
    ConfigDiarizacion,
    ConfigEjecucion,
    ConfigErrores,
    ConfigIdioma,
    ConfigLimpieza,
    ConfigNotas,
    ConfigRutas,
    ConfigSalida,
    categorias_por_defecto,
)


@pytest.fixture()
def categorias_de_prueba() -> tuple[CategoriaNota, ...]:
    """Las mismas categorías de ejemplo, con un override de modelo en una de
    ellas (para probar que el override de categoría gana sobre el general)."""
    from dataclasses import replace

    base = categorias_por_defecto()
    return tuple(
        replace(c, modelo="modelo-de-prueba-alternativo") if c.id == "nota_personal" else c
        for c in base
    )


@pytest.fixture()
def cfg(tmp_path: Path, categorias_de_prueba) -> Config:
    """Config apuntando a un árbol de directorios temporal (nunca a datos reales)."""
    base = tmp_path / "audios"
    base.mkdir()
    (base / "processed").mkdir()
    (base / "transcription" / "sent").mkdir(parents=True)

    rutas = ConfigRutas(
        base=base,
        # None = sin comprobación de montaje: no estorba en los tests.
        punto_montaje=None,
        procesados=base / "processed",
        transcripciones=base / "transcription",
        enviados=base / "transcription" / "sent",
        fallidos=base / "failed",
        scratch=tmp_path / "scratch",
        fichero_lock=tmp_path / "scratch" / "aura.lock",
        fichero_log=tmp_path / "estado" / "transcribe.log",
        fichero_estado=tmp_path / "estado" / "estado.json",
    )
    return Config(
        rutas=rutas,
        audio=ConfigAudio(segundos_estabilidad=0.0),
        asr=ConfigAsr(),
        idioma=ConfigIdioma(
            permitidos=("es", "gl"),
            idioma_alternativo="gl",
            confundibles_con_alternativo=("gl", "pt"),
            umbral_alternativo=0.5,
            idiomas_sin_alineamiento=("gl",),
        ),
        diarizacion=ConfigDiarizacion(),
        limpieza=ConfigLimpieza(),
        salida=ConfigSalida(),
        errores=ConfigErrores(),
        # Sin aislamiento: estos tests sustituyen el ASR con dobles en memoria,
        # que un subproceso real no vería (y que dispararía la GPU de verdad).
        # El camino aislado tiene sus propios tests en test_worker.py.
        ejecucion=ConfigEjecucion(aislar_subproceso=False),
        notas=ConfigNotas(
            activo=True,
            vault=tmp_path / "vault",
            modelo="modelo-de-prueba",
            categorias=categorias_de_prueba,
        ),
    )
