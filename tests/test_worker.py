"""Tests del aislamiento en subproceso: orden, serialización y fallos.

Nada de esto toca la GPU, ni Drive, ni lanza un proceso de verdad: `subprocess.run`
se sustituye por un doble. Lo que se comprueba es el contrato entre padre e hijo.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from aura_transcribe import pipeline as modulo_pipeline
from aura_transcribe import worker as modulo_worker
from aura_transcribe.config import Config, ConfigEjecucion
from aura_transcribe.discovery import AudioPendiente
from aura_transcribe.pipeline import (
    CENTINELA_RESULTADO,
    ErrorSubproceso,
    Opciones,
    ResultadoAudio,
    ejecutar,
    opciones_a_dict,
    opciones_desde_dict,
    resultado_a_dict,
    resultado_desde_dict,
    ruta_config_efectiva,
)


@pytest.fixture()
def sin_modelos(monkeypatch):
    """Dobles deterministas de ffprobe/ffmpeg/ASR/diarización (nada de GPU)."""
    from aura_transcribe import asr as modulo_asr
    from aura_transcribe.merge import Turno
    from aura_transcribe.output import Segmento

    def falso_normalizar(origen, destino, frecuencia=16000):
        Path(destino).parent.mkdir(parents=True, exist_ok=True)
        Path(destino).write_bytes(b"RIFFfalso")
        return Path(destino)

    def falso_transcribir(cfg, ruta_wav, *, idioma_cli=None, ruta_original=None):
        return modulo_asr.ResultadoAsr(
            segmentos=[Segmento(inicio=0.0, fin=2.0, texto="Hola qué tal.")],
            idioma_detectado="es",
            confianza_idioma=0.93,
            idioma_usado=idioma_cli or "es",
            hubo_alineamiento=True,
        )

    monkeypatch.setattr(modulo_pipeline, "duracion_segundos", lambda ruta: 24.08)
    monkeypatch.setattr(modulo_pipeline, "normalizar", falso_normalizar)
    monkeypatch.setattr(modulo_asr, "transcribir", falso_transcribir)
    monkeypatch.setattr(
        "aura_transcribe.diarize.diarizar",
        lambda cfg, ruta_wav: [Turno(0.0, 2.0, "SPEAKER_00")],
        raising=False,
    )


@pytest.fixture()
def cfg_aislado(cfg: Config) -> Config:
    """El cfg de siempre, pero con el aislamiento en subproceso activado."""
    return replace(cfg, ejecucion=ConfigEjecucion(aislar_subproceso=True))


def _audio(cfg: Config, nombre: str = "aura_prueba.wav") -> Path:
    ruta = cfg.rutas.base / nombre
    ruta.write_bytes(b"x" * 2048)
    viejo = time.time() - 3600
    os.utime(ruta, (viejo, viejo))
    return ruta


def _pendiente(ruta: Path) -> AudioPendiente:
    st = ruta.stat()
    return AudioPendiente(ruta=ruta, tamano=st.st_size, mtime=st.st_mtime)


# ---------------------------------------------------------------------------
# Construcción de la orden
# ---------------------------------------------------------------------------


def test_orden_worker_lleva_interprete_modulo_config_audio_y_opciones(cfg: Config):
    audio = _audio(cfg)
    opciones = Opciones(idioma="gl", ruta_config=Path("/ruta/mi.toml"))

    orden = modulo_pipeline._orden_worker(_pendiente(audio), opciones)

    assert orden[0] == sys.executable
    assert orden[1:3] == ["-m", "aura_transcribe.worker"]
    assert orden[orden.index("--config") + 1] == "/ruta/mi.toml"
    assert orden[orden.index("--audio") + 1] == str(audio)

    carga = json.loads(orden[orden.index("--opciones") + 1])
    assert carga["idioma"] == "gl"
    # El hijo nunca vuelve a aislar: sería un subproceso por subproceso.
    assert carga["sin_aislamiento"] is True
    assert "--verboso" not in orden


def test_orden_worker_propaga_el_verboso(cfg: Config):
    audio = _audio(cfg)
    orden = modulo_pipeline._orden_worker(_pendiente(audio), Opciones(verboso=True))
    assert "--verboso" in orden


def test_ruta_config_efectiva_usa_el_entorno_o_el_por_defecto(monkeypatch):
    from aura_transcribe.config import RUTA_CONFIG_POR_DEFECTO

    assert ruta_config_efectiva(Opciones(ruta_config=Path("/x.toml"))) == Path("/x.toml")

    monkeypatch.setenv("AURA_CONFIG", "/desde/entorno.toml")
    assert ruta_config_efectiva(Opciones()) == Path("/desde/entorno.toml")

    monkeypatch.delenv("AURA_CONFIG", raising=False)
    assert ruta_config_efectiva(Opciones()) == RUTA_CONFIG_POR_DEFECTO


def test_entorno_worker_mete_el_src_en_pythonpath():
    entorno = modulo_pipeline._entorno_worker()
    raiz = str(Path(modulo_pipeline.__file__).resolve().parents[1])
    assert raiz in entorno["PYTHONPATH"].split(os.pathsep)


# ---------------------------------------------------------------------------
# Serialización de opciones y resultado
# ---------------------------------------------------------------------------


def test_opciones_van_y_vuelven_intactas():
    opciones = Opciones(
        fichero=Path("/tmp/uno.wav"),
        forzar=True,
        sin_diarizacion=True,
        idioma="gl",
        conservar_audio=True,
        dry_run=False,
        limite=3,
        ruta_config=Path("/tmp/mi.toml"),
        verboso=True,
    )
    vuelta = opciones_desde_dict(json.loads(json.dumps(opciones_a_dict(opciones))))

    # Lo único que cambia a propósito es sin_aislamiento (el hijo ya está aislado).
    assert vuelta == replace(opciones, sin_aislamiento=True)
    assert isinstance(vuelta.fichero, Path)
    assert isinstance(vuelta.ruta_config, Path)


def test_opciones_vacias_van_y_vuelven():
    vuelta = opciones_desde_dict(json.loads(json.dumps(opciones_a_dict(Opciones()))))
    assert vuelta.fichero is None
    assert vuelta.ruta_config is None
    assert vuelta.idioma is None
    assert vuelta.limite is None


def test_opciones_desde_dict_ignora_claves_desconocidas():
    vuelta = opciones_desde_dict({"idioma": "es", "algo_del_futuro": 42})
    assert vuelta.idioma == "es"


def test_resultado_va_y_vuelve_intacto():
    resultado = ResultadoAudio(
        audio=Path("/tmp/uno.wav"),
        ok=True,
        json_salida=Path("/tmp/uno.json"),
        segundos=12.5,
        hablantes=4,
        segmentos=166,
    )
    vuelta = resultado_desde_dict(json.loads(json.dumps(resultado_a_dict(resultado))))
    assert vuelta == resultado
    assert isinstance(vuelta.audio, Path)
    assert isinstance(vuelta.json_salida, Path)


def test_resultado_sin_json_de_salida():
    vuelta = resultado_desde_dict(
        resultado_a_dict(ResultadoAudio(audio=Path("/tmp/x.wav"), ok=False, error="uy"))
    )
    assert vuelta.json_salida is None
    assert vuelta.error == "uy"


def test_extraer_carga_coge_la_linea_centinela_entre_ruido():
    salida = (
        "esto es ruido de alguna librería\n"
        f'{CENTINELA_RESULTADO} {{"ok": true, "audio": "/tmp/a.wav"}}\n'
    )
    assert modulo_pipeline._extraer_carga(salida) == {"ok": True, "audio": "/tmp/a.wav"}


def test_extraer_carga_devuelve_none_si_no_hay_nada_util():
    assert modulo_pipeline._extraer_carga("") is None
    assert modulo_pipeline._extraer_carga(None) is None
    assert modulo_pipeline._extraer_carga("solo ruido\n") is None
    assert modulo_pipeline._extraer_carga(f"{CENTINELA_RESULTADO} no-es-json") is None


def test_motivo_salida_explica_codigos_y_senales():
    assert "código 1" in modulo_pipeline._motivo_salida(1)
    assert "SIGKILL" in modulo_pipeline._motivo_salida(-signal.SIGKILL)
    assert "OOM" in modulo_pipeline._motivo_salida(-signal.SIGKILL)
    assert "SIGSEGV" in modulo_pipeline._motivo_salida(-signal.SIGSEGV)


# ---------------------------------------------------------------------------
# Camino feliz y caminos de fallo, con subprocess simulado
# ---------------------------------------------------------------------------


def _simular_subprocess(monkeypatch, *, returncode: int, stdout: str) -> list[list[str]]:
    """Sustituye subprocess.run y devuelve la lista de órdenes ejecutadas."""
    ordenes: list[list[str]] = []

    def falso_run(orden, **kwargs):
        ordenes.append(list(orden))
        # El padre solo captura stdout; stderr se hereda a propósito.
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is None
        return subprocess.CompletedProcess(orden, returncode, stdout=stdout, stderr=None)

    monkeypatch.setattr(modulo_pipeline.subprocess, "run", falso_run)
    return ordenes


def test_subproceso_ok_se_cuenta_como_procesado(cfg_aislado: Config, monkeypatch):
    audio = _audio(cfg_aislado)
    carga = resultado_a_dict(
        ResultadoAudio(
            audio=audio,
            ok=True,
            json_salida=cfg_aislado.rutas.transcripciones / "aura_prueba.json",
            segundos=9.9,
            hablantes=2,
            segmentos=17,
        )
    )
    ordenes = _simular_subprocess(
        monkeypatch,
        returncode=0,
        stdout=f"{CENTINELA_RESULTADO} {json.dumps(carga)}\n",
    )

    resultado = ejecutar(cfg_aislado, Opciones())

    assert len(ordenes) == 1, "un subproceso por audio"
    assert ordenes[0][1:3] == ["-m", "aura_transcribe.worker"]
    assert len(resultado.procesados) == 1
    assert resultado.procesados[0].segmentos == 17
    assert not resultado.fallidos
    assert not cfg_aislado.rutas.fichero_estado.exists() or json.loads(
        cfg_aislado.rutas.fichero_estado.read_text(encoding="utf-8")
    ) == {}


def test_subproceso_que_falla_anota_reintento_y_deja_el_audio(
    cfg_aislado: Config, monkeypatch
):
    """Código de salida != 0 y stdout vacío: el clásico proceso muerto por OOM."""
    audio = _audio(cfg_aislado)
    _simular_subprocess(monkeypatch, returncode=1, stdout="")

    resultado = ejecutar(cfg_aislado, Opciones())

    assert len(resultado.fallidos) == 1
    assert audio.exists(), "el audio nunca se pierde"
    assert not (cfg_aislado.rutas.transcripciones / "aura_prueba.json").exists()

    estado = json.loads(cfg_aislado.rutas.fichero_estado.read_text(encoding="utf-8"))
    assert estado["aura_prueba.wav"]["intentos"] == 1
    assert "código 1" in estado["aura_prueba.wav"]["ultimo_error"]


def test_subproceso_muerto_por_senal_es_un_fallo_normal(
    cfg_aislado: Config, monkeypatch
):
    audio = _audio(cfg_aislado)
    _simular_subprocess(monkeypatch, returncode=-signal.SIGKILL, stdout="")

    resultado = ejecutar(cfg_aislado, Opciones())

    assert len(resultado.fallidos) == 1
    assert audio.exists()
    estado = json.loads(cfg_aislado.rutas.fichero_estado.read_text(encoding="utf-8"))
    assert estado["aura_prueba.wav"]["intentos"] == 1


def test_subproceso_que_devuelve_error_propaga_el_traceback(
    cfg_aislado: Config, monkeypatch
):
    _audio(cfg_aislado)
    carga = {
        "ok": False,
        "error": "RuntimeError: CUDA failed with error out of memory",
        "traceback": "Traceback (most recent call last):\n  ...boom...",
    }
    _simular_subprocess(
        monkeypatch,
        returncode=1,
        stdout=f"{CENTINELA_RESULTADO} {json.dumps(carga)}\n",
    )

    with pytest.raises(ErrorSubproceso) as excinfo:
        modulo_pipeline._procesar_en_subproceso(
            cfg_aislado, _pendiente(cfg_aislado.rutas.base / "aura_prueba.wav"), Opciones()
        )
    assert "boom" in str(excinfo.value)


def test_tras_agotar_reintentos_el_subproceso_fallido_va_a_failed(
    cfg_aislado: Config, monkeypatch
):
    cfg_aislado = replace(
        cfg_aislado, errores=replace(cfg_aislado.errores, intentos_maximos=2)
    )
    audio = _audio(cfg_aislado)
    _simular_subprocess(monkeypatch, returncode=1, stdout="")

    ejecutar(cfg_aislado, Opciones())
    assert audio.exists()
    ejecutar(cfg_aislado, Opciones())

    assert not audio.exists()
    assert (cfg_aislado.rutas.fallidos / "aura_prueba.wav").is_file()
    error = cfg_aislado.rutas.fallidos / "aura_prueba.wav.error.txt"
    assert "subproceso" in error.read_text(encoding="utf-8")


def test_un_fallo_no_impide_procesar_los_siguientes(cfg_aislado: Config, monkeypatch):
    _audio(cfg_aislado, "a.wav")
    _audio(cfg_aislado, "b.wav")

    llamadas: list[list[str]] = []

    def falso_run(orden, **kwargs):
        llamadas.append(list(orden))
        ruta = Path(orden[orden.index("--audio") + 1])
        if ruta.name == "a.wav":
            return subprocess.CompletedProcess(orden, 1, stdout="", stderr=None)
        carga = resultado_a_dict(ResultadoAudio(audio=ruta, ok=True, segmentos=3))
        return subprocess.CompletedProcess(
            orden, 0, stdout=f"{CENTINELA_RESULTADO} {json.dumps(carga)}\n", stderr=None
        )

    monkeypatch.setattr(modulo_pipeline.subprocess, "run", falso_run)

    resultado = ejecutar(cfg_aislado, Opciones())

    assert len(llamadas) == 2
    assert len(resultado.fallidos) == 1
    assert len(resultado.procesados) == 1


def test_dry_run_no_lanza_subprocesos(cfg_aislado: Config, monkeypatch):
    audio = _audio(cfg_aislado)
    ordenes = _simular_subprocess(monkeypatch, returncode=0, stdout="")

    ejecutar(cfg_aislado, Opciones(dry_run=True))

    assert ordenes == []
    assert audio.exists()


def test_sin_aislamiento_no_lanza_subprocesos(cfg_aislado: Config, monkeypatch, sin_modelos):
    _audio(cfg_aislado)
    ordenes = _simular_subprocess(monkeypatch, returncode=0, stdout="")

    resultado = ejecutar(cfg_aislado, Opciones(sin_aislamiento=True))

    assert ordenes == []
    assert len(resultado.procesados) == 1
    assert (cfg_aislado.rutas.transcripciones / "aura_prueba.json").is_file()


# ---------------------------------------------------------------------------
# El worker en sí (llamando a main() en proceso, sin lanzar nada)
# ---------------------------------------------------------------------------


def test_worker_emite_el_resultado_por_stdout(cfg: Config, tmp_path, monkeypatch, capsys):
    audio = _audio(cfg)
    esperado = ResultadoAudio(audio=audio, ok=True, segundos=1.0, segmentos=5, hablantes=2)

    monkeypatch.setattr(modulo_worker, "cargar_config", lambda ruta: cfg)
    monkeypatch.setattr(
        modulo_worker, "procesar_audio", lambda c, p, o: esperado
    )

    codigo = modulo_worker.main(
        [
            "--config", str(tmp_path / "mi.toml"),
            "--audio", str(audio),
            "--opciones", json.dumps(opciones_a_dict(Opciones(idioma="gl"))),
        ]
    )

    assert codigo == 0
    carga = modulo_pipeline._extraer_carga(capsys.readouterr().out)
    assert resultado_desde_dict(carga) == esperado


def test_worker_devuelve_codigo_distinto_de_cero_y_traceback(
    cfg: Config, tmp_path, monkeypatch, capsys
):
    audio = _audio(cfg)

    def revienta(*args, **kwargs):
        raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch.setattr(modulo_worker, "cargar_config", lambda ruta: cfg)
    monkeypatch.setattr(modulo_worker, "procesar_audio", revienta)

    codigo = modulo_worker.main(
        [
            "--config", str(tmp_path / "mi.toml"),
            "--audio", str(audio),
            "--opciones", json.dumps(opciones_a_dict(Opciones())),
        ]
    )

    assert codigo != 0
    carga = modulo_pipeline._extraer_carga(capsys.readouterr().out)
    assert carga["ok"] is False
    assert "out of memory" in carga["error"]
    assert "Traceback" in carga["traceback"]


def test_worker_no_pide_el_lock(cfg: Config, tmp_path, monkeypatch, capsys):
    """El padre ya tiene el flock: si el hijo lo pidiera se bloquearía a sí mismo.

    flock() se asocia a la descripción de fichero abierta, así que un segundo
    intento -incluso desde el mismo proceso- chocaría con el primero. Aquí se
    coge el lock como hace `ejecutar_con_lock()` y se comprueba que el worker
    trabaja igual: la prueba de que no lo vuelve a pedir.
    """
    from aura_transcribe.filesystem import lock_exclusivo

    audio = _audio(cfg)
    esperado = ResultadoAudio(audio=audio, ok=True)

    monkeypatch.setattr(modulo_worker, "cargar_config", lambda ruta: cfg)
    monkeypatch.setattr(modulo_worker, "procesar_audio", lambda c, p, o: esperado)

    with lock_exclusivo(cfg.rutas.fichero_lock):
        codigo = modulo_worker.main(
            [
                "--config", str(tmp_path / "mi.toml"),
                "--audio", str(audio),
                "--opciones", json.dumps(opciones_a_dict(Opciones())),
            ]
        )

    assert codigo == 0
    assert resultado_desde_dict(
        modulo_pipeline._extraer_carga(capsys.readouterr().out)
    ) == esperado


def test_el_worker_no_menciona_el_lock_en_su_codigo():
    """Guardia barata contra reintroducir el interbloqueo por descuido."""
    fuente = Path(modulo_worker.__file__).read_text(encoding="utf-8")
    codigo = "\n".join(
        linea for linea in fuente.splitlines() if not linea.strip().startswith("#")
    )
    # Se descarta el docstring del módulo, que sí habla del lock a propósito.
    codigo = codigo.split('"""', 2)[-1]
    assert "lock_exclusivo" not in codigo
    assert "ejecutar_con_lock" not in codigo
