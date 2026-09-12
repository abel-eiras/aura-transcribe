"""Fusión de transcripción y diarización, y limpieza de segmentos.

Corrige lo que se veía mal en los JSON de junio (4 hablantes en un clip de 24 s):
  * tope de hablantes configurable (además del que se pide a pyannote),
  * fusión de segmentos consecutivos del mismo hablante separados por poco tiempo,
  * descarte de segmentos minúsculos sin texto útil.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

from .output import Segmento

log = logging.getLogger("aura.merge")

# Texto que no aporta nada: solo signos de puntuación, puntos suspensivos, etc.
_SOLO_RUIDO = re.compile(r"^[\s\.\,\-\–\—\¿\?\¡\!\:\;\"\'\«\»\(\)\[\]…]*$")


@dataclass(frozen=True)
class Turno:
    """Un turno de habla devuelto por la diarización."""

    inicio: float
    fin: float
    hablante: str

    @property
    def duracion(self) -> float:
        return max(0.0, self.fin - self.inicio)


def solapamiento(a_ini: float, a_fin: float, b_ini: float, b_fin: float) -> float:
    """Segundos de solape entre dos intervalos (0 si no se tocan)."""
    return max(0.0, min(a_fin, b_fin) - max(a_ini, b_ini))


def _hablante_por_solape(
    inicio: float, fin: float, turnos: list[Turno]
) -> tuple[str | None, float]:
    """Hablante con más solape temporal con [inicio, fin]."""
    acumulado: dict[str, float] = defaultdict(float)
    for turno in turnos:
        if turno.fin <= inicio:
            continue
        if turno.inicio >= fin:
            break  # los turnos vienen ordenados por inicio
        acumulado[turno.hablante] += solapamiento(inicio, fin, turno.inicio, turno.fin)

    if not acumulado:
        return None, 0.0
    mejor = max(acumulado.items(), key=lambda par: par[1])
    return (mejor[0], mejor[1]) if mejor[1] > 0 else (None, 0.0)


def _hablante_mas_cercano(punto: float, turnos: list[Turno]) -> str | None:
    """Hablante del turno más próximo en el tiempo (reserva si no hay solape)."""
    if not turnos:
        return None
    def distancia(turno: Turno) -> float:
        if turno.inicio <= punto <= turno.fin:
            return 0.0
        return min(abs(turno.inicio - punto), abs(turno.fin - punto))

    return min(turnos, key=distancia).hablante


def asignar_hablantes(
    segmentos: list[Segmento], turnos: list[Turno], *, usar_palabras: bool = True
) -> list[Segmento]:
    """Asigna hablante a cada segmento (y a cada palabra si las hay).

    Con alineamiento forzado la asignación se hace por palabra y el hablante del
    segmento es el que más tiempo de palabra acumula: mucho menos ruido que
    asignar por segmento entero.
    """
    if not turnos:
        return segmentos

    turnos = sorted(turnos, key=lambda t: (t.inicio, t.fin))

    for seg in segmentos:
        palabras_con_tiempo = [
            p
            for p in (seg.palabras or [])
            if p.get("start") is not None and p.get("end") is not None
        ]

        if usar_palabras and palabras_con_tiempo:
            pesos: dict[str, float] = defaultdict(float)
            for palabra in palabras_con_tiempo:
                hablante, solape = _hablante_por_solape(
                    float(palabra["start"]), float(palabra["end"]), turnos
                )
                if hablante is None:
                    hablante = _hablante_mas_cercano(
                        (float(palabra["start"]) + float(palabra["end"])) / 2.0, turnos
                    )
                    solape = 0.0
                palabra["speaker"] = hablante
                if hablante:
                    duracion = max(
                        1e-3, float(palabra["end"]) - float(palabra["start"])
                    )
                    pesos[hablante] += max(solape, duracion * 0.25)
            if pesos:
                seg.hablante = max(pesos.items(), key=lambda par: par[1])[0]
                continue

        hablante, _ = _hablante_por_solape(seg.inicio, seg.fin, turnos)
        if hablante is None:
            hablante = _hablante_mas_cercano((seg.inicio + seg.fin) / 2.0, turnos)
        seg.hablante = hablante

    # Los segmentos sin hablante (silencios raros) heredan el del vecino anterior.
    ultimo: str | None = None
    for seg in segmentos:
        if seg.hablante:
            ultimo = seg.hablante
        elif ultimo:
            seg.hablante = ultimo
    return segmentos


def limitar_hablantes(segmentos: list[Segmento], maximo: int) -> list[Segmento]:
    """Reduce a `maximo` hablantes quedándose con los que más tiempo hablan.

    Los segmentos de los hablantes descartados se reasignan al hablante
    dominante más próximo en el tiempo. Es una red de seguridad por encima del
    max_speakers que ya se le pide a pyannote.
    """
    if maximo <= 0:
        return segmentos

    tiempos: dict[str, float] = defaultdict(float)
    for seg in segmentos:
        if seg.hablante:
            tiempos[seg.hablante] += seg.duracion
    if len(tiempos) <= maximo:
        return segmentos

    principales = {
        nombre
        for nombre, _ in sorted(tiempos.items(), key=lambda par: par[1], reverse=True)[
            :maximo
        ]
    }
    log.info(
        "Se detectaron %d hablantes; se recortan a %d (%s)",
        len(tiempos),
        maximo,
        ", ".join(sorted(principales)),
    )

    indices_principales = [
        i for i, seg in enumerate(segmentos) if seg.hablante in principales
    ]
    for i, seg in enumerate(segmentos):
        if seg.hablante in principales:
            continue
        if not indices_principales:
            seg.hablante = None
            continue
        centro = (seg.inicio + seg.fin) / 2.0
        vecino = min(
            indices_principales,
            key=lambda j: abs(
                ((segmentos[j].inicio + segmentos[j].fin) / 2.0) - centro
            ),
        )
        seg.hablante = segmentos[vecino].hablante
    return segmentos


def descartar_minusculos(
    segmentos: list[Segmento], duracion_minima: float
) -> list[Segmento]:
    """Elimina segmentos por debajo de `duracion_minima` sin texto útil."""
    resultado = []
    for seg in segmentos:
        texto = (seg.texto or "").strip()
        if seg.duracion < duracion_minima and (
            not texto or _SOLO_RUIDO.match(texto) or len(texto) <= 2
        ):
            log.debug(
                "Descartado segmento minúsculo %.2f-%.2f (%r)", seg.inicio, seg.fin, texto
            )
            continue
        if not texto:
            continue
        seg.texto = texto
        resultado.append(seg)
    return resultado


def fusionar_consecutivos(
    segmentos: list[Segmento], hueco_maximo_ms: float
) -> list[Segmento]:
    """Une segmentos seguidos del mismo hablante separados por < hueco_maximo_ms."""
    if not segmentos:
        return []
    hueco = hueco_maximo_ms / 1000.0
    fusionados: list[Segmento] = [segmentos[0]]

    for seg in segmentos[1:]:
        previo = fusionados[-1]
        mismo_hablante = previo.hablante == seg.hablante
        if mismo_hablante and (seg.inicio - previo.fin) <= hueco:
            previo.fin = max(previo.fin, seg.fin)
            previo.texto = f"{previo.texto.rstrip()} {seg.texto.lstrip()}".strip()
            previo.palabras = (previo.palabras or []) + (seg.palabras or [])
        else:
            fusionados.append(seg)
    return fusionados


def renumerar_hablantes(segmentos: list[Segmento]) -> list[Segmento]:
    """Renombra los hablantes a SPEAKER_00, SPEAKER_01... por orden de aparición."""
    mapa: dict[str, str] = {}
    for seg in segmentos:
        if seg.hablante and seg.hablante not in mapa:
            mapa[seg.hablante] = f"SPEAKER_{len(mapa):02d}"
    for seg in segmentos:
        if seg.hablante:
            seg.hablante = mapa[seg.hablante]
        for palabra in seg.palabras or []:
            if palabra.get("speaker"):
                palabra["speaker"] = mapa.get(palabra["speaker"], palabra["speaker"])
    return segmentos


def limpiar(
    segmentos: list[Segmento],
    *,
    max_hablantes: int,
    hueco_fusion_ms: float,
    duracion_minima_s: float,
) -> list[Segmento]:
    """Aplica toda la limpieza de la sección 5.6 del plan, en orden."""
    segmentos = descartar_minusculos(segmentos, duracion_minima_s)
    segmentos = limitar_hablantes(segmentos, max_hablantes)
    segmentos = fusionar_consecutivos(segmentos, hueco_fusion_ms)
    segmentos = renumerar_hablantes(segmentos)
    return segmentos
