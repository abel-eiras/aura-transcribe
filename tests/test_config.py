"""Tests de la carga de configuración y del token de Hugging Face."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura_transcribe import config as modulo_config
from aura_transcribe.config import (
    BASE_POR_DEFECTO,
    ErrorConfig,
    cargar_config,
    cargar_token_hf,
    exigir_token_hf,
)


def _limpiar_entorno(monkeypatch):
    for clave in (
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AURA_BASE_DIR",
        "AURA_MODELO",
        "AURA_DISPOSITIVO",
        "AURA_PUNTO_MONTAJE",
    ):
        monkeypatch.delenv(clave, raising=False)


def test_config_inexistente_usa_los_valores_por_defecto(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    cfg = cargar_config(tmp_path / "no-existe.toml")
    assert cfg.rutas.base == BASE_POR_DEFECTO
    assert cfg.rutas.punto_montaje is None
    assert cfg.asr.modelo == "large-v3"
    assert cfg.errores.intentos_maximos == 3
    assert cfg.ejecucion.aislar_subproceso is True
    # Las tres categorías de ejemplo, presentes cuando no se sobreescriben.
    assert {c.id for c in cfg.notas.categorias} == {
        "reunion",
        "visita_cliente",
        "nota_personal",
    }


def test_override_por_entorno(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    monkeypatch.setenv("AURA_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("AURA_MODELO", "large-v3-turbo")
    cfg = cargar_config(tmp_path / "no-existe.toml")
    assert cfg.rutas.base == tmp_path
    assert cfg.rutas.transcripciones == tmp_path / "transcription"
    assert cfg.asr.modelo == "large-v3-turbo"


def test_toml_propio(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    toml = tmp_path / "mi.toml"
    toml.write_text(
        '[rutas]\nbase = "%s"\n[asr]\nmodelo = "small"\n[diarizacion]\nmax_hablantes = 2\n'
        % tmp_path,
        encoding="utf-8",
    )
    cfg = cargar_config(toml)
    assert cfg.asr.modelo == "small"
    assert cfg.diarizacion.max_hablantes == 2


def test_punto_montaje_solo_se_activa_si_se_configura(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    toml = tmp_path / "mi.toml"
    toml.write_text(
        '[rutas]\nbase = "%s"\npunto_montaje = "/mnt/lo-que-sea"\n' % tmp_path,
        encoding="utf-8",
    )
    cfg = cargar_config(toml)
    assert cfg.rutas.punto_montaje == Path("/mnt/lo-que-sea")


def test_aislar_subproceso_se_puede_apagar_desde_el_toml(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    toml = tmp_path / "mi.toml"
    toml.write_text("[ejecucion]\naislar_subproceso = false\n", encoding="utf-8")
    assert cargar_config(toml).ejecucion.aislar_subproceso is False


def test_categorias_de_notas_personalizadas_reemplazan_las_de_ejemplo(
    tmp_path: Path, monkeypatch
):
    _limpiar_entorno(monkeypatch)
    toml = tmp_path / "mi.toml"
    toml.write_text(
        """
[[notas.categorias]]
id = "diario"
etiqueta = "Diario"
carpeta = "Diario"
tag = "diario"
criterio = "reflexión personal del día"
prompt_redaccion = "Redacta una entrada de diario a partir de la transcripción."
""",
        encoding="utf-8",
    )
    cfg = cargar_config(toml)
    assert [c.id for c in cfg.notas.categorias] == ["diario"]
    assert cfg.notas.categorias[0].carpeta == "Diario"


def test_categoria_incompleta_lanza_error_de_config(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    toml = tmp_path / "mi.toml"
    toml.write_text(
        '[[notas.categorias]]\nid = "diario"\netiqueta = "Diario"\n',
        encoding="utf-8",
    )
    with pytest.raises(ErrorConfig):
        cargar_config(toml)


def test_categorias_con_id_repetido_lanza_error_de_config(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    toml = tmp_path / "mi.toml"
    plantilla = (
        '[[notas.categorias]]\nid = "x"\netiqueta = "X"\ncarpeta = "X"\ntag = "x"\n'
        'criterio = "c"\nprompt_redaccion = "p"\n'
    )
    toml.write_text(plantilla * 2, encoding="utf-8")
    with pytest.raises(ErrorConfig):
        cargar_config(toml)


def test_token_desde_el_entorno(monkeypatch):
    _limpiar_entorno(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_de_prueba")
    assert cargar_token_hf() == "hf_de_prueba"


def test_token_desde_el_fichero_de_config(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    fichero = tmp_path / "env"
    fichero.write_text(
        "# comentario\nexport HF_TOKEN='hf_del_fichero'\nOTRA=cosa\n", encoding="utf-8"
    )
    monkeypatch.setattr(modulo_config, "RUTA_ENV_USUARIO", fichero)
    assert cargar_token_hf() == "hf_del_fichero"


def test_sin_token_el_mensaje_explica_los_pasos(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    monkeypatch.setattr(modulo_config, "RUTA_ENV_USUARIO", tmp_path / "no-existe")
    assert cargar_token_hf() is None
    with pytest.raises(ErrorConfig) as excinfo:
        exigir_token_hf()
    mensaje = str(excinfo.value)
    assert "huggingface.co" in mensaje
    assert "speaker-diarization-community-1" in mensaje
    assert "--no-diarization" in mensaje
