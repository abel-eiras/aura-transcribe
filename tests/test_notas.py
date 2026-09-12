"""Tests de la generación de notas de Obsidian (clasificación y redacción
mockeadas: aquí no se prueba la calidad del LLM, solo el cableado)."""

from __future__ import annotations

import json

from aura_transcribe.notas import (
    OpcionesNotas,
    _fecha_desde,
    ejecutar_notas,
    generar_nota,
)


def _documento(nombre: str, *, idioma: str = "es") -> dict:
    return {
        "metadata": {
            "source_file": f"{nombre}.wav",
            "processed_at": "2026-09-08T12:00:00Z",
            "duration_seconds": 1920.0,
            "language_used": idioma,
            "num_speakers_detected": 3,
        },
        "segments": [],
        "full_text": "SPEAKER_00: hola\nSPEAKER_01: hola",
    }


def _escribir_json(cfg, carpeta, nombre: str, *, idioma: str = "es") -> None:
    ruta = carpeta / f"{nombre}.json"
    ruta.write_text(json.dumps(_documento(nombre, idioma=idioma)), encoding="utf-8")


def test_generar_nota_reunion(cfg, monkeypatch):
    _escribir_json(cfg, cfg.rutas.transcripciones, "20260908_125746")
    ruta_json = cfg.rutas.transcripciones / "20260908_125746.json"

    monkeypatch.setattr("aura_transcribe.notas.clasificar", lambda *a, **k: "reunion")
    monkeypatch.setattr(
        "aura_transcribe.notas.completar_chat",
        lambda **_: (
            "## Asistentes\nSPEAKER_00, SPEAKER_01\n"
            "## Puntos tratados\nAlgo\n"
            "## Decisiones tomadas\nNinguna\n"
            "## Tareas\nNinguna\n"
            "## Pendientes\nNinguno"
        ),
    )

    destino = generar_nota(cfg, ruta_json, "clave")

    assert destino is not None
    assert destino.parent.name == "Reuniones"
    assert destino.name == "2026-09-08-20260908_125746.md"
    contenido = destino.read_text(encoding="utf-8")
    assert "tipo: reunion" in contenido
    assert "fecha: 2026-09-08" in contenido
    assert 'audio_origen: "20260908_125746.wav"' in contenido
    assert "tags: [aura, reunion]" in contenido
    assert "## Puntos tratados" in contenido


def test_categoria_con_override_de_modelo_lo_usa_en_vez_del_general(cfg, monkeypatch):
    _escribir_json(cfg, cfg.rutas.transcripciones, "personal", idioma="gl")
    ruta_json = cfg.rutas.transcripciones / "personal.json"

    monkeypatch.setattr(
        "aura_transcribe.notas.clasificar", lambda *a, **k: "nota_personal"
    )
    capturado = {}

    def _completar_falso(**kwargs):
        capturado["modelo"] = kwargs["modelo"]
        return "## Contenido\nx\n## Tareas o recordatorios\nNinguna"

    monkeypatch.setattr("aura_transcribe.notas.completar_chat", _completar_falso)

    destino = generar_nota(cfg, ruta_json, "clave")

    categoria_personal = next(
        c for c in cfg.notas.categorias if c.id == "nota_personal"
    )
    assert destino.parent.name == "Notas personales"
    assert categoria_personal.modelo  # la fixture le puso un override
    assert capturado["modelo"] == categoria_personal.modelo


def test_se_omite_si_ya_existe_una_nota_para_ese_json(cfg, monkeypatch):
    _escribir_json(cfg, cfg.rutas.transcripciones, "20260908_125746")
    ruta_json = cfg.rutas.transcripciones / "20260908_125746.json"

    destino_previo = cfg.notas.vault / "Reuniones" / "2026-09-08-20260908_125746.md"
    destino_previo.parent.mkdir(parents=True)
    destino_previo.write_text("ya existía", encoding="utf-8")

    llamadas = {"n": 0}

    def _clasificar_falso(*a, **k):
        llamadas["n"] += 1
        return "reunion"

    monkeypatch.setattr("aura_transcribe.notas.clasificar", _clasificar_falso)

    resultado = generar_nota(cfg, ruta_json, "clave")

    assert resultado is None
    assert llamadas["n"] == 0  # se detecta por nombre de fichero, sin reclasificar
    assert destino_previo.read_text(encoding="utf-8") == "ya existía"


def test_force_regenera_aunque_exista(cfg, monkeypatch):
    _escribir_json(cfg, cfg.rutas.transcripciones, "20260908_125746")
    ruta_json = cfg.rutas.transcripciones / "20260908_125746.json"

    destino_previo = cfg.notas.vault / "Reuniones" / "2026-09-08-20260908_125746.md"
    destino_previo.parent.mkdir(parents=True)
    destino_previo.write_text("versión vieja", encoding="utf-8")

    monkeypatch.setattr("aura_transcribe.notas.clasificar", lambda *a, **k: "reunion")
    monkeypatch.setattr(
        "aura_transcribe.notas.completar_chat", lambda **_: "## Asistentes\nnuevo"
    )

    destino = generar_nota(cfg, ruta_json, "clave", forzar=True)

    assert destino == destino_previo
    assert "nuevo" in destino.read_text(encoding="utf-8")


def test_fecha_desde_usa_el_nombre_del_audio_si_sigue_el_patron():
    documento = {"metadata": {"processed_at": "2026-01-02T03:04:05Z"}}
    assert _fecha_desde(documento, "20260908_125746.json") == "2026-09-08"


def test_fecha_desde_usa_processed_at_si_no_hay_patron_en_el_nombre():
    documento = {"metadata": {"processed_at": "2026-01-02T03:04:05Z"}}
    assert _fecha_desde(documento, "nota-suelta.json") == "2026-01-02"


def test_ejecutar_notas_sin_backlog_ignora_sent(cfg, monkeypatch):
    _escribir_json(cfg, cfg.rutas.transcripciones, "a")
    _escribir_json(cfg, cfg.rutas.enviados, "b")

    monkeypatch.setattr("aura_transcribe.notas.clasificar", lambda *a, **k: "reunion")
    monkeypatch.setattr("aura_transcribe.notas.completar_chat", lambda **_: "cuerpo")

    resultado = ejecutar_notas(cfg, OpcionesNotas(backlog=False), "clave")

    assert len(resultado.generadas) == 1
    assert resultado.generadas[0].stem.endswith("-a")


def test_ejecutar_notas_backlog_recorre_transcription_y_sent(cfg, monkeypatch):
    _escribir_json(cfg, cfg.rutas.transcripciones, "a")
    _escribir_json(cfg, cfg.rutas.enviados, "b")

    monkeypatch.setattr("aura_transcribe.notas.clasificar", lambda *a, **k: "reunion")
    monkeypatch.setattr("aura_transcribe.notas.completar_chat", lambda **_: "cuerpo")

    resultado = ejecutar_notas(cfg, OpcionesNotas(backlog=True), "clave")

    assert len(resultado.generadas) == 2


def test_audio_sin_texto_se_omite_sin_llamar_al_modelo(cfg, monkeypatch):
    ruta_json = cfg.rutas.transcripciones / "vacio.json"
    ruta_json.write_text(
        json.dumps({"metadata": {"source_file": "vacio.wav"}, "full_text": ""}),
        encoding="utf-8",
    )

    llamadas = {"n": 0}

    def _clasificar_falso(*a, **k):
        llamadas["n"] += 1
        return "reunion"

    monkeypatch.setattr("aura_transcribe.notas.clasificar", _clasificar_falso)

    resultado = generar_nota(cfg, ruta_json, "clave")

    assert resultado is None
    assert llamadas["n"] == 0


def test_reclasificar_con_force_borra_la_nota_de_la_carpeta_vieja(cfg, monkeypatch):
    _escribir_json(cfg, cfg.rutas.transcripciones, "20260908_125106")
    ruta_json = cfg.rutas.transcripciones / "20260908_125106.json"

    vieja = cfg.notas.vault / "Reuniones" / "2026-09-08-20260908_125106.md"
    vieja.parent.mkdir(parents=True)
    vieja.write_text("nota vieja, categoría equivocada", encoding="utf-8")

    # Reclasifica a una categoría distinta a la de la nota ya existente.
    monkeypatch.setattr(
        "aura_transcribe.notas.clasificar", lambda *a, **k: "nota_personal"
    )
    monkeypatch.setattr(
        "aura_transcribe.notas.completar_chat",
        lambda **_: "## Contenido\nx\n## Tareas o recordatorios\nNinguna",
    )

    destino = generar_nota(cfg, ruta_json, "clave", forzar=True)

    assert destino.parent.name == "Notas personales"
    assert not vieja.exists()  # la nota vieja se borra, no queda duplicada
    assert destino.exists()


def test_audio_sin_texto_borra_una_nota_previa_si_la_hubiera(cfg):
    ruta_json = cfg.rutas.transcripciones / "vacio.json"
    ruta_json.write_text(
        json.dumps({"metadata": {"source_file": "vacio.wav"}, "full_text": ""}),
        encoding="utf-8",
    )
    vieja = cfg.notas.vault / "Reuniones" / "2026-09-08-vacio.md"
    vieja.parent.mkdir(parents=True)
    vieja.write_text("nota generada por error sobre un audio vacío", encoding="utf-8")

    resultado = generar_nota(cfg, ruta_json, "clave", forzar=True)

    assert resultado is None
    assert not vieja.exists()


def test_ejecutar_notas_registra_fallos_sin_interrumpir_el_resto(cfg, monkeypatch):
    (cfg.rutas.transcripciones / "roto.json").write_text("no es json", encoding="utf-8")
    _escribir_json(cfg, cfg.rutas.transcripciones, "b")

    monkeypatch.setattr("aura_transcribe.notas.clasificar", lambda *a, **k: "reunion")
    monkeypatch.setattr("aura_transcribe.notas.completar_chat", lambda **_: "cuerpo")

    resultado = ejecutar_notas(cfg, OpcionesNotas(), "clave")

    assert len(resultado.fallidas) == 1
    assert resultado.fallidas[0][0].name == "roto.json"
    assert len(resultado.generadas) == 1
