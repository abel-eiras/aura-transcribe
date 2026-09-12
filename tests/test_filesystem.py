"""Tests de las operaciones de sistema de ficheros."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from aura_transcribe.filesystem import (
    ErrorLock,
    ErrorMontaje,
    asegurar_directorios,
    escribir_json_atomico,
    escribir_texto_atomico,
    es_estable,
    esta_montado,
    exigir_montaje,
    guardar_estado,
    leer_estado,
    lock_exclusivo,
    mover_atomico,
)


def test_escritura_atomica_de_texto(tmp_path: Path):
    destino = tmp_path / "sub" / "fichero.txt"
    escribir_texto_atomico(destino, "hola")
    assert destino.read_text(encoding="utf-8") == "hola"
    # No queda basura temporal.
    assert list(destino.parent.glob(".*tmp*")) == []


def test_escritura_atomica_sobreescribe(tmp_path: Path):
    destino = tmp_path / "f.txt"
    escribir_texto_atomico(destino, "v1")
    escribir_texto_atomico(destino, "v2")
    assert destino.read_text(encoding="utf-8") == "v2"


def test_escritura_json_conserva_acentos(tmp_path: Path):
    destino = tmp_path / "f.json"
    escribir_json_atomico(destino, {"texto": "año así ñ"})
    crudo = destino.read_text(encoding="utf-8")
    assert "año así ñ" in crudo
    assert json.loads(crudo)["texto"] == "año así ñ"


def test_escritura_atomica_no_deja_temporal_si_falla(tmp_path: Path):
    destino = tmp_path / "f.json"

    class NoSerializable:
        pass

    with pytest.raises(TypeError):
        escribir_json_atomico(destino, {"x": NoSerializable()})
    assert not destino.exists()
    assert list(tmp_path.glob(".*tmp*")) == []


def test_mover_atomico(tmp_path: Path):
    origen = tmp_path / "a.wav"
    origen.write_bytes(b"datos")
    destino_dir = tmp_path / "processed"
    destino = mover_atomico(origen, destino_dir)
    assert destino == destino_dir / "a.wav"
    assert destino.read_bytes() == b"datos"
    assert not origen.exists()


def test_mover_atomico_no_pisa_lo_existente(tmp_path: Path):
    destino_dir = tmp_path / "processed"
    destino_dir.mkdir()
    (destino_dir / "a.wav").write_bytes(b"viejo")
    origen = tmp_path / "a.wav"
    origen.write_bytes(b"nuevo")

    destino = mover_atomico(origen, destino_dir)

    assert destino.name == "a__1.wav"
    assert (destino_dir / "a.wav").read_bytes() == b"viejo"
    assert destino.read_bytes() == b"nuevo"


def test_lock_impide_dos_ejecuciones(tmp_path: Path):
    ruta = tmp_path / "aura.lock"
    with lock_exclusivo(ruta):
        # Un segundo flock desde otro proceso debe fallar.
        import subprocess
        import sys

        codigo = (
            "import sys;"
            "sys.path.insert(0, %r);" % str(Path(__file__).resolve().parents[1] / "src")
            + "from aura_transcribe.filesystem import lock_exclusivo, ErrorLock;"
            "import pathlib;"
            "\ntry:\n"
            "    with lock_exclusivo(pathlib.Path(%r)):\n" % str(ruta)
            + "        print('OBTENIDO')\n"
            "except ErrorLock:\n"
            "    print('BLOQUEADO')\n"
        )
        salida = subprocess.run(
            [sys.executable, "-c", codigo], capture_output=True, text=True
        )
        assert "BLOQUEADO" in salida.stdout, salida


def test_lock_se_libera_al_salir(tmp_path: Path):
    ruta = tmp_path / "aura.lock"
    with lock_exclusivo(ruta):
        pass
    with lock_exclusivo(ruta):
        pass  # No debe lanzar.


def test_estabilidad(tmp_path: Path):
    fichero = tmp_path / "a.wav"
    fichero.write_bytes(b"x" * 10)
    assert not es_estable(fichero, segundos=60)
    viejo = time.time() - 300
    os.utime(fichero, (viejo, viejo))
    assert es_estable(fichero, segundos=60)


def test_estabilidad_fichero_vacio(tmp_path: Path):
    fichero = tmp_path / "a.wav"
    fichero.write_bytes(b"")
    viejo = time.time() - 300
    os.utime(fichero, (viejo, viejo))
    assert not es_estable(fichero, segundos=60)


def test_estabilidad_fichero_inexistente(tmp_path: Path):
    assert not es_estable(tmp_path / "no-existe.wav", segundos=1)


def test_montaje(tmp_path: Path):
    assert esta_montado(Path("/"))
    with pytest.raises(ErrorMontaje):
        exigir_montaje(tmp_path / "no-montado-jamas")


def test_estado_ida_y_vuelta(tmp_path: Path):
    ruta = tmp_path / "estado.json"
    assert leer_estado(ruta) == {}
    guardar_estado(ruta, {"a.wav": {"intentos": 2}})
    assert leer_estado(ruta)["a.wav"]["intentos"] == 2


def test_estado_corrupto_no_revienta(tmp_path: Path):
    ruta = tmp_path / "estado.json"
    ruta.write_text("{esto no es json", encoding="utf-8")
    assert leer_estado(ruta) == {}


def test_asegurar_directorios(tmp_path: Path):
    a = tmp_path / "x" / "y"
    asegurar_directorios(a, a)  # idempotente
    assert a.is_dir()
