"""Tests de la coordinación opcional con un sincronizador externo.

Si la carpeta de trabajo la sincroniza también otro proceso (por ejemplo un
`rclone bisync` propio, protegido con `flock -n` sobre un fichero), el
pipeline puede coger ESE MISMO lock, pero solo durante la ventana en la que
muta el árbol (escribir el JSON + mover el audio). Ver `[rutas]
lock_sincronizacion` en config.example.toml; por defecto está desactivado
(None) y el pipeline no coordina con nada externo.

Aquí no hay ni systemd ni rclone: el lock ficticio es un fichero temporal que
los tests bloquean con `flock` desde el propio proceso. Vale igual, porque los
flock van asociados a la descripción de fichero abierta, no al proceso: dos
`os.open()` del mismo fichero en el mismo proceso se bloquean entre sí.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from aura_transcribe.config import (
    LOCK_SINCRONIZACION_POR_DEFECTO,
    Config,
    cargar_config,
)
from aura_transcribe.filesystem import ErrorLockSincronizacion, lock_sincronizacion
from aura_transcribe.pipeline import Opciones, ejecutar

from test_pipeline import _audio, sin_modelos  # noqa: F401 - fixture reutilizada


class LockAjeno:
    """Simula al rclone del usuario: coge el flock y lo suelta cuando se le dice."""

    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)
        self.fd: int | None = None

    def coger(self) -> None:
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(self.ruta), os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def soltar(self) -> None:
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


@pytest.fixture()
def lock_ajeno(tmp_path: Path):
    ajeno = LockAjeno(tmp_path / "sync-falso.lock")
    try:
        yield ajeno
    finally:
        ajeno.soltar()


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------


def _limpiar_entorno(monkeypatch):
    for clave in ("AURA_BASE_DIR", "AURA_LOCK_SINCRONIZACION", "AURA_PUNTO_MONTAJE"):
        monkeypatch.delenv(clave, raising=False)


def test_sin_config_toml_la_coordinacion_esta_desactivada(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    cfg = cargar_config(tmp_path / "no-existe.toml")
    assert cfg.rutas.lock_sincronizacion == LOCK_SINCRONIZACION_POR_DEFECTO
    assert cfg.rutas.lock_sincronizacion is None
    assert cfg.ejecucion.espera_lock_sincronizacion == 120.0
    # Es un lock DISTINTO del que usa el pipeline para no pisarse consigo mismo.
    assert cfg.rutas.lock_sincronizacion != cfg.rutas.fichero_lock


def test_lock_vacio_desactiva_la_coordinacion(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    ruta = tmp_path / "config.toml"
    ruta.write_text('[rutas]\nlock_sincronizacion = ""\n', encoding="utf-8")
    cfg = cargar_config(ruta)
    assert cfg.rutas.lock_sincronizacion is None


def test_lock_y_espera_personalizados(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    ruta = tmp_path / "config.toml"
    ruta.write_text(
        '[rutas]\nlock_sincronizacion = "/tmp/otro.lock"\n'
        "[ejecucion]\nespera_lock_sincronizacion = 7.5\n",
        encoding="utf-8",
    )
    cfg = cargar_config(ruta)
    assert cfg.rutas.lock_sincronizacion == Path("/tmp/otro.lock")
    assert cfg.ejecucion.espera_lock_sincronizacion == 7.5


def test_override_por_entorno(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    monkeypatch.setenv("AURA_LOCK_SINCRONIZACION", str(tmp_path / "x.lock"))
    cfg = cargar_config(tmp_path / "no-existe.toml")
    assert cfg.rutas.lock_sincronizacion == tmp_path / "x.lock"

    monkeypatch.setenv("AURA_LOCK_SINCRONIZACION", "")
    cfg = cargar_config(tmp_path / "no-existe.toml")
    assert cfg.rutas.lock_sincronizacion is None


def test_cambiar_la_base_por_entorno_no_activa_ningun_lock(tmp_path: Path, monkeypatch):
    _limpiar_entorno(monkeypatch)
    monkeypatch.setenv("AURA_BASE_DIR", str(tmp_path))
    cfg = cargar_config(tmp_path / "no-existe.toml")
    assert cfg.rutas.lock_sincronizacion == LOCK_SINCRONIZACION_POR_DEFECTO


# ---------------------------------------------------------------------------
# El contextmanager
# ---------------------------------------------------------------------------


def test_lock_libre_se_coge_al_instante(tmp_path: Path):
    inicio = time.monotonic()
    with lock_sincronizacion(tmp_path / "s.lock", espera_maxima=5.0) as fd:
        assert fd is not None
    assert time.monotonic() - inicio < 1.0


def test_ruta_none_no_hace_nada(tmp_path: Path):
    with lock_sincronizacion(None, espera_maxima=0.0) as fd:
        assert fd is None


def test_se_crea_el_fichero_de_lock_si_no_existe(tmp_path: Path):
    ruta = tmp_path / "sub" / "s.lock"
    with lock_sincronizacion(ruta, espera_maxima=1.0):
        pass
    assert ruta.is_file()


def test_espera_a_que_el_otro_suelte(lock_ajeno: LockAjeno):
    lock_ajeno.coger()
    threading.Timer(0.4, lock_ajeno.soltar).start()

    inicio = time.monotonic()
    with lock_sincronizacion(lock_ajeno.ruta, espera_maxima=10.0, intervalo=0.05):
        esperado = time.monotonic() - inicio

    # Esperó de verdad (no se coló) y terminó en cuanto el otro soltó.
    assert 0.3 <= esperado < 5.0


def test_si_no_lo_suelta_nadie_salta_el_plazo(lock_ajeno: LockAjeno):
    lock_ajeno.coger()
    inicio = time.monotonic()
    with pytest.raises(ErrorLockSincronizacion):
        with lock_sincronizacion(lock_ajeno.ruta, espera_maxima=0.3, intervalo=0.05):
            pytest.fail("no debería haberse entrado en la ventana de mutación")
    assert time.monotonic() - inicio >= 0.3


def test_el_lock_se_suelta_aunque_reviente_el_cuerpo(tmp_path: Path):
    ruta = tmp_path / "s.lock"
    with pytest.raises(ZeroDivisionError):
        with lock_sincronizacion(ruta, espera_maxima=1.0):
            1 / 0
    # Si no se hubiera soltado, esto se quedaría colgado hasta el plazo.
    with lock_sincronizacion(ruta, espera_maxima=0.5):
        pass


# ---------------------------------------------------------------------------
# El pipeline bajo el lock
# ---------------------------------------------------------------------------


def _cfg_con_lock(cfg: Config, ruta: Path, espera: float) -> Config:
    return replace(
        cfg,
        rutas=replace(cfg.rutas, lock_sincronizacion=ruta),
        ejecucion=replace(cfg.ejecucion, espera_lock_sincronizacion=espera),
    )


def test_el_pipeline_espera_a_que_termine_rclone(
    cfg: Config, sin_modelos, lock_ajeno: LockAjeno  # noqa: F811
):
    """Con el lock ocupado, la pasada espera y luego escribe con normalidad."""
    cfg = _cfg_con_lock(cfg, lock_ajeno.ruta, espera=10.0)
    audio = _audio(cfg)

    lock_ajeno.coger()
    threading.Timer(0.4, lock_ajeno.soltar).start()

    inicio = time.monotonic()
    resultado = ejecutar(cfg, Opciones())
    tardanza = time.monotonic() - inicio

    assert len(resultado.procesados) == 1
    assert not resultado.fallidos
    assert tardanza >= 0.3  # esperó al lock, no se coló
    assert (cfg.rutas.transcripciones / "aura_prueba.json").is_file()
    assert (cfg.rutas.procesados / audio.name).is_file()
    assert not audio.exists()


def test_si_el_lock_no_se_libera_el_audio_se_queda_y_se_anota_reintento(
    cfg: Config, sin_modelos, lock_ajeno: LockAjeno  # noqa: F811
):
    """Agotar el plazo es un fallo normal del audio: nada se toca, se reintenta."""
    import json

    cfg = _cfg_con_lock(cfg, lock_ajeno.ruta, espera=0.3)
    audio = _audio(cfg)
    lock_ajeno.coger()  # nadie lo suelta

    resultado = ejecutar(cfg, Opciones())

    assert not resultado.procesados
    assert len(resultado.fallidos) == 1
    assert "sincronización" in resultado.fallidos[0].error

    # El audio sigue en su sitio y no hay JSON a medias.
    assert audio.is_file()
    assert not (cfg.rutas.procesados / audio.name).exists()
    assert not (cfg.rutas.transcripciones / "aura_prueba.json").exists()
    assert list(cfg.rutas.transcripciones.glob(".*tmp*")) == []

    estado = json.loads(cfg.rutas.fichero_estado.read_text(encoding="utf-8"))
    assert estado[audio.name]["intentos"] == 1


def test_con_la_coordinacion_desactivada_no_se_mira_ningun_lock(
    cfg: Config, sin_modelos, lock_ajeno: LockAjeno  # noqa: F811
):
    """lock_sincronizacion = None: se procesa aunque el lock esté cogido."""
    cfg = _cfg_con_lock(cfg, None, espera=0.3)
    _audio(cfg)
    lock_ajeno.coger()

    resultado = ejecutar(cfg, Opciones())

    assert len(resultado.procesados) == 1
    assert not resultado.fallidos


def test_el_lock_solo_se_coge_al_final_no_durante_el_asr(
    cfg: Config, monkeypatch, sin_modelos, lock_ajeno: LockAjeno  # noqa: F811
):
    """La ventana de mutación empieza DESPUÉS del ASR y de la diarización.

    Si el lock se cogiera al principio, una transcripción de 20 minutos dejaría
    al usuario sin sincronizar todo ese rato. Se comprueba que durante el ASR el
    lock sigue libre (otro proceso puede cogerlo) y que solo se toma al escribir.
    """
    from aura_transcribe import asr as modulo_asr

    libre_durante_el_asr = {}
    original = modulo_asr.transcribir

    def espiar(cfg_, ruta_wav, **kwargs):
        sonda = LockAjeno(lock_ajeno.ruta)
        try:
            sonda.coger()
            libre_durante_el_asr["si"] = True
        except OSError:
            libre_durante_el_asr["si"] = False
        finally:
            sonda.soltar()
        return original(cfg_, ruta_wav, **kwargs)

    monkeypatch.setattr(modulo_asr, "transcribir", espiar)

    cfg = _cfg_con_lock(cfg, lock_ajeno.ruta, espera=5.0)
    _audio(cfg)
    resultado = ejecutar(cfg, Opciones())

    assert len(resultado.procesados) == 1
    assert libre_durante_el_asr == {"si": True}
