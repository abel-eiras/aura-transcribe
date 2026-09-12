"""Carga de la configuración del pipeline.

Orden de precedencia (de menor a mayor):
    valores por defecto  <  config.toml  <  variables de entorno  <  opciones de la CLI
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

# Ruta del config.toml que acompaña al proyecto (dos niveles por encima del paquete).
RUTA_CONFIG_POR_DEFECTO = Path(__file__).resolve().parents[2] / "config.toml"

# Raíz del repo (tres niveles por encima del paquete): donde vive el vault de
# Obsidian por defecto. No tiene nada que ver con [rutas].base, que es la
# carpeta donde aparecen los audios nuevos.
RUTA_REPO_POR_DEFECTO = Path(__file__).resolve().parents[3]

# Fichero donde el usuario guarda sus claves (nunca en el repo).
RUTA_ENV_USUARIO = Path.home() / ".config" / "aura-transcribe" / "env"

# Carpeta de trabajo por defecto si no hay config.toml ni override de entorno.
BASE_POR_DEFECTO = Path.home() / "AudioNotes" / "aura"

# Lock opcional para coordinarse con un script externo de sincronización
# (p. ej. un `rclone bisync` sobre la misma carpeta). Vacío/None = desactivado.
LOCK_SINCRONIZACION_POR_DEFECTO: Path | None = None


@dataclass(frozen=True)
class ConfigRutas:
    base: Path
    # None = no se comprueba ningún punto de montaje antes de tocar `base`.
    # Rellénalo solo si `base` vive en un volumen que puede aparecer
    # desmontado (una unidad de red, un mount de rclone, un disco externo...):
    # ver [rutas] punto_montaje en config.example.toml.
    punto_montaje: Path | None
    procesados: Path
    transcripciones: Path
    enviados: Path
    fallidos: Path
    scratch: Path
    fichero_lock: Path
    fichero_log: Path
    fichero_estado: Path
    # Lock COMPARTIDO con un proceso externo (p. ej. un sync de Drive/Nextcloud
    # por rclone); None desactiva la coordinación. No confundir con
    # `fichero_lock`, que es el lock propio del pipeline para que dos pasadas
    # suyas no se pisen. Son dos locks distintos con dueños distintos.
    lock_sincronizacion: Path | None = None


@dataclass(frozen=True)
class ConfigAudio:
    extensiones: tuple[str, ...] = (".wav", ".ogg", ".m4a", ".mp3", ".opus")
    segundos_estabilidad: float = 20.0
    frecuencia_destino: int = 16000


@dataclass(frozen=True)
class ConfigAsr:
    modelo: str = "large-v3"
    tipo_computo: str = "int8_float16"
    dispositivo: str = "cuda"
    tamano_lote: int = 4
    usar_vad: bool = True


@dataclass(frozen=True)
class ConfigIdioma:
    """Detección de idioma restringida: nunca "libre".

    Solo se contemplan dos idiomas: `por_defecto` y, opcionalmente, uno
    `idioma_alternativo` (p. ej. una segunda lengua cooficial). Cualquier otra
    detección se descarta y se usa `por_defecto`, para que un audio corto y
    ruidoso no acabe transcrito en un idioma cualquiera.
    """

    por_defecto: str = "es"
    permitidos: tuple[str, ...] = ("es",)
    # Idioma que se acepta como alternativa al de por defecto. "" lo desactiva:
    # todo lo que no sea `por_defecto` se descarta.
    idioma_alternativo: str = ""
    # Códigos que el detector puede confundir con `idioma_alternativo` (p. ej.
    # dos lenguas próximas suelen confundirse entre sí en los primeros segundos).
    confundibles_con_alternativo: tuple[str, ...] = ()
    umbral_alternativo: float = 0.5
    segundos_deteccion: float = 45.0
    alinear: bool = True
    idiomas_sin_alineamiento: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigDiarizacion:
    activa: bool = True
    modelo: str = "pyannote/speaker-diarization-community-1"
    max_hablantes: int = 4
    min_hablantes: int = 1
    reserva_cpu: bool = True


@dataclass(frozen=True)
class ConfigLimpieza:
    hueco_fusion_ms: float = 800.0
    duracion_minima_s: float = 0.3


@dataclass(frozen=True)
class ConfigSalida:
    escribir_txt: bool = False
    incluir_palabras: bool = False


@dataclass(frozen=True)
class ConfigErrores:
    intentos_maximos: int = 3


@dataclass(frozen=True)
class ConfigEjecucion:
    """Cómo se ejecuta cada audio dentro de una pasada.

    `aislar_subproceso` existe por una razón medida, no teórica: en una GPU con
    poca VRAM la fragmentación se acumula a lo largo de un proceso de vida
    larga y un lote de varios audios puede reventar con
    `CUDA failed with error out of memory` en el cuarto o quinto fichero, pese a
    liberar la caché entre etapas. El mismo audio, relanzado solo en un proceso
    nuevo, no tiene problema: la única forma fiable de devolver TODA la VRAM al
    sistema es que el proceso muera. En una GPU con VRAM sobrada se puede
    desactivar sin riesgo con `--sin-aislamiento` (algo más rápido por lote).
    """

    aislar_subproceso: bool = True
    # Segundos que se espera al lock de sincronización externo antes de dar el
    # audio por fallido.
    espera_lock_sincronizacion: float = 120.0


@dataclass(frozen=True)
class CategoriaNota:
    """Un tipo de nota que el pipeline puede generar en el vault de Obsidian.

    Totalmente definida por configuración: no hay ninguna categoría
    "especial" en el código. `criterio` es lo único que ve el clasificador
    para elegir entre categorías; `prompt_redaccion` es el prompt de sistema
    completo (rol, formato de salida, idioma de la nota si aplica) que se le
    pasa al modelo para redactarla.
    """

    id: str
    etiqueta: str
    carpeta: str
    tag: str
    criterio: str
    prompt_redaccion: str
    # Override opcional de [notas].modelo, solo para esta categoría (p. ej.
    # un modelo con mejor soporte de un idioma minoritario).
    modelo: str = ""


def categorias_por_defecto() -> tuple[CategoriaNota, ...]:
    """Tres categorías de ejemplo, pensadas para notas de voz de trabajo.

    Son solo un punto de partida: cualquier proyecto puede sustituirlas por
    completo en `[[notas.categorias]]` de config.toml. Ver config.example.toml
    y la sección "Tipos de nota" del README.
    """
    return (
        CategoriaNota(
            id="reunion",
            etiqueta="Reunión",
            carpeta="Reuniones",
            tag="reunion",
            criterio=(
                "registro más o menos formal, varios hablantes, temas de "
                "trabajo/organización"
            ),
            prompt_redaccion=(
                "Redactas el acta de una reunión a partir de la transcripción "
                "literal que te paso (con hablantes SPEAKER_00, SPEAKER_01...). "
                "No es una transcripción ni un resumen genérico: para cada tema "
                "tratado cuenta la sustancia —de qué se habló, qué se acordó y "
                "qué quedó sin acordar o pendiente de decidir—, aunque el tema "
                "haya ocupado media hora de conversación. No reproduzcas la "
                "conversación intervención por intervención; sí conserva los "
                "detalles concretos que importan (cifras, nombres, plazos, "
                "condiciones). Responde en Markdown con EXACTAMENTE estas "
                "secciones, en este orden, y nada más:\n\n"
                "## Asistentes\n## Puntos tratados\n## Decisiones tomadas\n"
                "## Tareas\n## Pendientes\n\n"
                "En «Asistentes» lista las etiquetas de hablante y, solo si el "
                "nombre real se menciona explícitamente en el audio, añádelo "
                "entre paréntesis. En «Puntos tratados» un punto por tema, con "
                "su desenlace (qué se acordó o no), no una crónica de cómo se "
                "discutió. En «Tareas» indica el responsable si se menciona. Si "
                "una sección no tiene contenido, escribe «Ninguna»."
            ),
        ),
        CategoriaNota(
            id="visita_cliente",
            etiqueta="Visita cliente",
            carpeta="Visitas clientes",
            tag="visita-cliente",
            criterio=(
                "registro conversacional/informal con un cliente; se distingue "
                "de una reunión por el tono, aunque el tema también sea de "
                "trabajo"
            ),
            prompt_redaccion=(
                "Redactas la minuta de una visita a un cliente a partir de la "
                "transcripción literal que te paso (hablantes SPEAKER_00, "
                "SPEAKER_01...), de registro conversacional. No es una "
                "transcripción ni un resumen genérico: para cada tema cuenta la "
                "sustancia —de qué se habló y en qué quedó—, con los detalles "
                "concretos que importan (cifras, plazos, condiciones), sin "
                "reproducir la conversación intervención por intervención. "
                "Responde en Markdown con EXACTAMENTE estas secciones, en este "
                "orden, y nada más:\n\n"
                "## Contexto\n## Temas tratados\n## Peticiones del cliente\n"
                "## Compromisos adquiridos\n## Seguimiento\n\n"
                "Si una sección no tiene contenido, escribe «Ninguno»."
            ),
        ),
        CategoriaNota(
            id="nota_personal",
            etiqueta="Nota personal",
            carpeta="Notas personales",
            tag="nota-personal",
            criterio="sin relación temática con trabajo ni clientes",
            prompt_redaccion=(
                "A partir de la transcripción literal de una nota de voz "
                "personal que te paso, redacta una nota estructurada. No es "
                "una transcripción ni un resumen genérico: para cada idea o "
                "tema recoge la sustancia y los detalles concretos que "
                "importan, sin reproducir el audio palabra por palabra. "
                "Responde en Markdown con EXACTAMENTE estas secciones, en este "
                "orden, y nada más:\n\n"
                "## Contenido\n## Tareas o recordatorios\n\n"
                "Si «Tareas o recordatorios» no tiene contenido, escribe "
                "«Ninguna»."
            ),
        ),
    )


@dataclass(frozen=True)
class ConfigNotas:
    """Generación de notas de Obsidian a partir de las transcripciones.

    Ver la sección "Notas de Obsidian" del README. `activo = false` por
    defecto: activarlo implica mandar el texto completo de las transcripciones
    a OpenRouter, y eso lo decide quien lo activa, no el valor por defecto.
    """

    activo: bool = False
    vault: Path = field(default_factory=lambda: RUTA_REPO_POR_DEFECTO / "vault")
    modelo: str = "deepseek/deepseek-v4-flash"
    categorias: tuple[CategoriaNota, ...] = field(default_factory=categorias_por_defecto)


@dataclass(frozen=True)
class Config:
    rutas: ConfigRutas
    audio: ConfigAudio = field(default_factory=ConfigAudio)
    asr: ConfigAsr = field(default_factory=ConfigAsr)
    idioma: ConfigIdioma = field(default_factory=ConfigIdioma)
    diarizacion: ConfigDiarizacion = field(default_factory=ConfigDiarizacion)
    limpieza: ConfigLimpieza = field(default_factory=ConfigLimpieza)
    salida: ConfigSalida = field(default_factory=ConfigSalida)
    errores: ConfigErrores = field(default_factory=ConfigErrores)
    ejecucion: ConfigEjecucion = field(default_factory=ConfigEjecucion)
    notas: ConfigNotas = field(default_factory=ConfigNotas)


class ErrorConfig(RuntimeError):
    """La configuración es inválida o falta algo imprescindible."""


def _leer_toml(ruta: Path) -> dict:
    if not ruta.is_file():
        return {}
    with ruta.open("rb") as fh:
        return tomllib.load(fh)


def _resolver(base: Path, valor: str) -> Path:
    """Resuelve `valor` contra `base` si es relativo."""
    ruta = Path(os.path.expanduser(valor))
    return ruta if ruta.is_absolute() else (base / ruta)


def _resolver_repo(valor: str) -> Path:
    """Como `_resolver()` pero contra la raíz del repo, no contra la carpeta de audios."""
    return _resolver(RUTA_REPO_POR_DEFECTO, valor)


def _rutas_desde_dict(datos: dict) -> ConfigRutas:
    base = Path(os.path.expanduser(datos.get("base", str(BASE_POR_DEFECTO))))
    punto_montaje_bruto = str(datos.get("punto_montaje", "") or "").strip()
    punto_montaje = (
        Path(os.path.expanduser(punto_montaje_bruto)) if punto_montaje_bruto else None
    )

    scratch_bruto = datos.get("scratch") or ""
    scratch = (
        Path(os.path.expanduser(scratch_bruto))
        if scratch_bruto
        else Path(tempfile.gettempdir()) / "aura-transcribe"
    )

    estado_base = Path.home() / ".local" / "state" / "aura-transcribe"

    lock_bruto = datos.get("fichero_lock") or ""
    log_bruto = datos.get("fichero_log") or ""
    estado_bruto = datos.get("fichero_estado") or ""

    sinc_bruto = str(datos.get("lock_sincronizacion") or "").strip()
    lock_sincronizacion = Path(os.path.expanduser(sinc_bruto)) if sinc_bruto else None

    return ConfigRutas(
        base=base,
        punto_montaje=punto_montaje,
        procesados=_resolver(base, datos.get("procesados", "processed")),
        transcripciones=_resolver(base, datos.get("transcripciones", "transcription")),
        enviados=_resolver(base, datos.get("enviados", "transcription/sent")),
        fallidos=_resolver(base, datos.get("fallidos", "failed")),
        scratch=scratch,
        fichero_lock=(
            Path(os.path.expanduser(lock_bruto)) if lock_bruto else scratch / "aura-transcribe.lock"
        ),
        fichero_log=(
            Path(os.path.expanduser(log_bruto)) if log_bruto else estado_base / "transcribe.log"
        ),
        fichero_estado=(
            Path(os.path.expanduser(estado_bruto)) if estado_bruto else estado_base / "estado.json"
        ),
        lock_sincronizacion=lock_sincronizacion,
    )


def _categoria_desde_dict(datos: dict, indice: int) -> CategoriaNota:
    faltantes = [
        clave
        for clave in ("id", "etiqueta", "carpeta", "tag", "criterio", "prompt_redaccion")
        if not str(datos.get(clave, "")).strip()
    ]
    if faltantes:
        raise ErrorConfig(
            f"[[notas.categorias]] #{indice + 1}: faltan campos obligatorios: "
            f"{', '.join(faltantes)}"
        )
    return CategoriaNota(
        id=str(datos["id"]).strip(),
        etiqueta=str(datos["etiqueta"]).strip(),
        carpeta=str(datos["carpeta"]).strip(),
        tag=str(datos["tag"]).strip(),
        criterio=str(datos["criterio"]).strip(),
        prompt_redaccion=str(datos["prompt_redaccion"]).strip(),
        modelo=str(datos.get("modelo", "")).strip(),
    )


def _categorias_desde_dict(datos_notas: dict) -> tuple[CategoriaNota, ...]:
    brutas = datos_notas.get("categorias")
    if not brutas:
        return categorias_por_defecto()

    categorias = tuple(
        _categoria_desde_dict(dict(c), i) for i, c in enumerate(brutas)
    )
    ids = [c.id for c in categorias]
    duplicados = {i for i in ids if ids.count(i) > 1}
    if duplicados:
        raise ErrorConfig(
            f"[[notas.categorias]]: id repetido: {', '.join(sorted(duplicados))}"
        )
    return categorias


def cargar_config(ruta_config: Path | None = None) -> Config:
    """Carga config.toml y aplica los overrides de entorno."""
    ruta = ruta_config or Path(
        os.environ.get("AURA_CONFIG", str(RUTA_CONFIG_POR_DEFECTO))
    )
    datos = _leer_toml(Path(ruta))

    cfg = Config(
        rutas=_rutas_desde_dict(datos.get("rutas", {})),
        audio=ConfigAudio(
            extensiones=tuple(
                str(e).lower()
                for e in datos.get("audio", {}).get(
                    "extensiones", ConfigAudio.extensiones
                )
            ),
            segundos_estabilidad=float(
                datos.get("audio", {}).get("segundos_estabilidad", 20.0)
            ),
            frecuencia_destino=int(
                datos.get("audio", {}).get("frecuencia_destino", 16000)
            ),
        ),
        asr=ConfigAsr(
            modelo=str(datos.get("asr", {}).get("modelo", "large-v3")),
            tipo_computo=str(datos.get("asr", {}).get("tipo_computo", "int8_float16")),
            dispositivo=str(datos.get("asr", {}).get("dispositivo", "cuda")),
            tamano_lote=int(datos.get("asr", {}).get("tamano_lote", 4)),
            usar_vad=bool(datos.get("asr", {}).get("usar_vad", True)),
        ),
        idioma=ConfigIdioma(
            por_defecto=str(datos.get("idioma", {}).get("por_defecto", "es")),
            permitidos=tuple(datos.get("idioma", {}).get("permitidos", ("es",))),
            idioma_alternativo=str(
                datos.get("idioma", {}).get("idioma_alternativo", "")
            ),
            confundibles_con_alternativo=tuple(
                datos.get("idioma", {}).get("confundibles_con_alternativo", ())
            ),
            umbral_alternativo=float(
                datos.get("idioma", {}).get("umbral_alternativo", 0.5)
            ),
            segundos_deteccion=float(
                datos.get("idioma", {}).get("segundos_deteccion", 45.0)
            ),
            alinear=bool(datos.get("idioma", {}).get("alinear", True)),
            idiomas_sin_alineamiento=tuple(
                datos.get("idioma", {}).get("idiomas_sin_alineamiento", ())
            ),
        ),
        diarizacion=ConfigDiarizacion(
            activa=bool(datos.get("diarizacion", {}).get("activa", True)),
            modelo=str(
                datos.get("diarizacion", {}).get(
                    "modelo", "pyannote/speaker-diarization-community-1"
                )
            ),
            max_hablantes=int(datos.get("diarizacion", {}).get("max_hablantes", 4)),
            min_hablantes=int(datos.get("diarizacion", {}).get("min_hablantes", 1)),
            reserva_cpu=bool(datos.get("diarizacion", {}).get("reserva_cpu", True)),
        ),
        limpieza=ConfigLimpieza(
            hueco_fusion_ms=float(datos.get("limpieza", {}).get("hueco_fusion_ms", 800)),
            duracion_minima_s=float(
                datos.get("limpieza", {}).get("duracion_minima_s", 0.3)
            ),
        ),
        salida=ConfigSalida(
            escribir_txt=bool(datos.get("salida", {}).get("escribir_txt", False)),
            incluir_palabras=bool(
                datos.get("salida", {}).get("incluir_palabras", False)
            ),
        ),
        errores=ConfigErrores(
            intentos_maximos=int(datos.get("errores", {}).get("intentos_maximos", 3))
        ),
        ejecucion=ConfigEjecucion(
            aislar_subproceso=bool(
                datos.get("ejecucion", {}).get("aislar_subproceso", True)
            ),
            espera_lock_sincronizacion=float(
                datos.get("ejecucion", {}).get("espera_lock_sincronizacion", 120.0)
            ),
        ),
        notas=ConfigNotas(
            activo=bool(datos.get("notas", {}).get("activo", False)),
            vault=_resolver_repo(datos.get("notas", {}).get("vault", "vault")),
            modelo=str(
                datos.get("notas", {}).get("modelo", "deepseek/deepseek-v4-flash")
            ),
            categorias=_categorias_desde_dict(datos.get("notas", {})),
        ),
    )

    return _aplicar_entorno(cfg)


def _aplicar_entorno(cfg: Config) -> Config:
    """Overrides por variables de entorno AURA_*."""
    base_env = os.environ.get("AURA_BASE_DIR")
    if base_env:
        base = Path(os.path.expanduser(base_env))
        cfg = replace(
            cfg,
            rutas=replace(
                cfg.rutas,
                base=base,
                procesados=base / "processed",
                transcripciones=base / "transcription",
                enviados=base / "transcription" / "sent",
                fallidos=base / "failed",
            ),
        )

    montaje_env = os.environ.get("AURA_PUNTO_MONTAJE")
    if montaje_env:
        cfg = replace(
            cfg, rutas=replace(cfg.rutas, punto_montaje=Path(montaje_env))
        )

    sinc_env = os.environ.get("AURA_LOCK_SINCRONIZACION")
    if sinc_env is not None:
        cfg = replace(
            cfg,
            rutas=replace(
                cfg.rutas,
                lock_sincronizacion=(
                    Path(os.path.expanduser(sinc_env)) if sinc_env.strip() else None
                ),
            ),
        )

    modelo_env = os.environ.get("AURA_MODELO")
    if modelo_env:
        cfg = replace(cfg, asr=replace(cfg.asr, modelo=modelo_env))

    dispositivo_env = os.environ.get("AURA_DISPOSITIVO")
    if dispositivo_env:
        cfg = replace(cfg, asr=replace(cfg.asr, dispositivo=dispositivo_env))

    return cfg


def _leer_clave_env_usuario(*nombres: str) -> str | None:
    """Busca cualquiera de `nombres` en RUTA_ENV_USUARIO (formato `CLAVE=valor`,
    admite `export`). None si el fichero no existe o no contiene ninguna."""
    if not RUTA_ENV_USUARIO.is_file():
        return None

    for linea in RUTA_ENV_USUARIO.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        if linea.startswith("export "):
            linea = linea[len("export ") :].strip()
        if "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        if clave.strip() in nombres:
            return valor.strip().strip("'\"") or None

    return None


def cargar_token_hf() -> str | None:
    """Devuelve el token de Hugging Face, o None si no está disponible.

    Se busca, por este orden:
      1. La variable de entorno HF_TOKEN / HUGGINGFACE_TOKEN.
      2. El fichero ~/.config/aura-transcribe/env (formato `CLAVE=valor`,
         admite `export`).

    El token nunca se registra en el log ni se escribe en el repositorio.
    """
    for clave in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        valor = os.environ.get(clave)
        if valor and valor.strip():
            return valor.strip()

    return _leer_clave_env_usuario(
        "HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"
    )


def cargar_token_openrouter() -> str | None:
    """Devuelve la clave de OpenRouter, o None si no está disponible.

    Mismo mecanismo que `cargar_token_hf()`: variable de entorno
    OPENROUTER_API_KEY, o ~/.config/aura-transcribe/env. Nunca se registra en
    el log ni se escribe en el repositorio.
    """
    valor = os.environ.get("OPENROUTER_API_KEY")
    if valor and valor.strip():
        return valor.strip()

    return _leer_clave_env_usuario("OPENROUTER_API_KEY")


MENSAJE_SIN_TOKEN = (
    "No se encontró el token de Hugging Face, imprescindible para la diarización.\n"
    "Pasos (una sola vez):\n"
    "  1. Crea una cuenta gratuita en https://huggingface.co\n"
    "  2. Acepta las condiciones de pyannote/speaker-diarization-community-1\n"
    "     y de pyannote/segmentation-3.0 (aprobación instantánea).\n"
    "  3. Crea un token de lectura en https://huggingface.co/settings/tokens\n"
    f"  4. Guárdalo como  HF_TOKEN=hf_...  en {RUTA_ENV_USUARIO}\n"
    "     (o expórtalo como variable de entorno HF_TOKEN).\n"
    "Mientras tanto puedes transcribir sin hablantes con  --no-diarization."
)


def exigir_token_hf() -> str:
    """Devuelve el token o lanza ErrorConfig con instrucciones claras."""
    token = cargar_token_hf()
    if not token:
        raise ErrorConfig(MENSAJE_SIN_TOKEN)
    return token


MENSAJE_SIN_TOKEN_OPENROUTER = (
    "No se encontró la clave de OpenRouter, imprescindible para generar notas "
    "(ver la sección \"Notas de Obsidian\" del README).\n"
    "Pasos (una sola vez):\n"
    "  1. Crea una cuenta en https://openrouter.ai\n"
    "  2. Genera una clave en https://openrouter.ai/keys\n"
    f"  3. Guárdala como  OPENROUTER_API_KEY=sk-or-...  en {RUTA_ENV_USUARIO}\n"
    "     (o expórtala como variable de entorno OPENROUTER_API_KEY).\n"
)


def exigir_token_openrouter() -> str:
    """Devuelve la clave o lanza ErrorConfig con instrucciones claras."""
    token = cargar_token_openrouter()
    if not token:
        raise ErrorConfig(MENSAJE_SIN_TOKEN_OPENROUTER)
    return token
