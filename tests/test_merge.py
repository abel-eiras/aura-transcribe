"""Tests de la fusión con la diarización y de la limpieza de segmentos."""

from __future__ import annotations

from aura_transcribe.merge import (
    Turno,
    asignar_hablantes,
    descartar_minusculos,
    fusionar_consecutivos,
    limitar_hablantes,
    limpiar,
    renumerar_hablantes,
    solapamiento,
)
from aura_transcribe.output import Segmento


def seg(inicio, fin, texto="hola", hablante=None, palabras=None):
    return Segmento(
        inicio=inicio, fin=fin, texto=texto, hablante=hablante, palabras=palabras or []
    )


def test_solapamiento():
    assert solapamiento(0, 2, 1, 3) == 1
    assert solapamiento(0, 1, 2, 3) == 0
    assert solapamiento(0, 5, 1, 2) == 1


def test_asignacion_por_segmento():
    segmentos = [seg(0, 2), seg(3, 5)]
    turnos = [Turno(0, 2.5, "A"), Turno(2.6, 6, "B")]
    asignar_hablantes(segmentos, turnos, usar_palabras=False)
    assert [s.hablante for s in segmentos] == ["A", "B"]


def test_asignacion_por_palabra_gana_a_la_de_segmento():
    # El segmento entero solapa más con A, pero casi todas sus palabras son de B.
    palabras = [
        {"word": "uno", "start": 0.0, "end": 0.2},
        {"word": "dos", "start": 2.1, "end": 2.5},
        {"word": "tres", "start": 2.6, "end": 3.0},
        {"word": "cuatro", "start": 3.1, "end": 3.9},
    ]
    segmentos = [seg(0, 4, palabras=palabras)]
    turnos = [Turno(0, 2.0, "A"), Turno(2.0, 4.0, "B")]
    asignar_hablantes(segmentos, turnos, usar_palabras=True)
    assert segmentos[0].hablante == "B"
    assert palabras[0]["speaker"] == "A"
    assert palabras[-1]["speaker"] == "B"


def test_asignacion_sin_solape_usa_el_turno_mas_cercano():
    segmentos = [seg(10, 11)]
    turnos = [Turno(0, 2, "A"), Turno(12, 14, "B")]
    asignar_hablantes(segmentos, turnos, usar_palabras=False)
    assert segmentos[0].hablante == "B"


def test_sin_turnos_no_toca_nada():
    segmentos = [seg(0, 1)]
    asignar_hablantes(segmentos, [], usar_palabras=False)
    assert segmentos[0].hablante is None


def test_fusion_de_consecutivos_del_mismo_hablante():
    segmentos = [
        seg(0.0, 1.0, "Hola", "A"),
        seg(1.2, 2.0, "qué tal", "A"),
        seg(2.1, 3.0, "bien", "B"),
    ]
    fusionados = fusionar_consecutivos(segmentos, hueco_maximo_ms=800)
    assert len(fusionados) == 2
    assert fusionados[0].texto == "Hola qué tal"
    assert fusionados[0].fin == 2.0
    assert fusionados[1].hablante == "B"


def test_no_se_fusiona_si_el_hueco_es_grande():
    segmentos = [seg(0, 1, "a", "A"), seg(5, 6, "b", "A")]
    assert len(fusionar_consecutivos(segmentos, hueco_maximo_ms=800)) == 2


def test_descarte_de_segmentos_minusculos():
    segmentos = [
        seg(0.0, 0.1, "..."),
        seg(0.2, 0.35, "e"),
        seg(1.0, 1.2, "Sí"),      # corto pero con texto de más de 2 caracteres... no
        seg(2.0, 4.0, "Frase larga de verdad"),
    ]
    limpios = descartar_minusculos(segmentos, duracion_minima=0.3)
    textos = [s.texto for s in limpios]
    assert "..." not in textos
    assert "e" not in textos
    assert "Frase larga de verdad" in textos


def test_no_se_descartan_segmentos_largos_aunque_sean_cortos_de_texto():
    segmentos = [seg(0.0, 2.0, "Sí")]
    assert len(descartar_minusculos(segmentos, duracion_minima=0.3)) == 1


def test_limite_de_hablantes():
    # Cuatro hablantes en un clip corto: se recorta a 2 (los que más hablan).
    segmentos = [
        seg(0, 5, "largo A", "A"),
        seg(5, 10, "largo B", "B"),
        seg(10, 10.2, "pio", "C"),
        seg(10.3, 10.5, "pio", "D"),
    ]
    limitar_hablantes(segmentos, maximo=2)
    assert {s.hablante for s in segmentos} == {"A", "B"}


def test_limite_no_hace_nada_si_ya_cabe():
    segmentos = [seg(0, 1, "x", "A"), seg(1, 2, "y", "B")]
    limitar_hablantes(segmentos, maximo=4)
    assert [s.hablante for s in segmentos] == ["A", "B"]


def test_renumeracion_por_orden_de_aparicion():
    segmentos = [seg(0, 1, "x", "SPEAKER_03"), seg(1, 2, "y", "SPEAKER_01")]
    renumerar_hablantes(segmentos)
    assert [s.hablante for s in segmentos] == ["SPEAKER_00", "SPEAKER_01"]


def test_limpieza_completa_reduce_el_ruido_de_junio():
    # Reproduce el patrón del JSON de junio: "Dime." / "Dime." / "Hola, nena."
    # del mismo hablante en 0,7 s, más un cuarto hablante espurio.
    segmentos = [
        seg(2.127, 4.41, "Vale, el widget parece funcionar.", "SPEAKER_00"),
        seg(5.832, 7.494, "Estamos grabando desde el widget.", "SPEAKER_00"),
        seg(7.514, 7.754, "Dime.", "SPEAKER_03"),
        seg(7.774, 7.954, "Dime.", "SPEAKER_03"),
        seg(7.974, 8.255, "Hola, nena.", "SPEAKER_03"),
    ]
    limpios = limpiar(
        segmentos, max_hablantes=4, hueco_fusion_ms=800, duracion_minima_s=0.3
    )
    # Los dos primeros NO se fusionan (1,42 s de hueco > 800 ms); los tres
    # "Dime./Dime./Hola, nena." sí, porque van seguidos y son del mismo hablante.
    assert len(limpios) == 3
    assert [s.hablante for s in limpios] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    assert limpios[2].texto == "Dime. Dime. Hola, nena."
    assert limpios[2].inicio == 7.514 and limpios[2].fin == 8.255
