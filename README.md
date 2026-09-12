# aura-transcribe

## El porqué

[Aura](https://github.com/abel-eiras/Aura) (la app hermana de este repo) ya
resuelve la mitad barata del cacharrito de marras: grabar. Pero para que eso
sirva de algo hace falta la otra mitad —transcribir, separar quién habla y
escribir una nota decente—, que es justo la parte que esos trastos te cobran
aparte, mes a mes, como si fuera brujería.

No lo es. Aquí tienes esa otra mitad, y corre casi entera en tu propio
ordenador. Casi entera, porque la redacción final de la nota sale por
OpenRouter — el porqué ya lo confesé en el README de Aura, así que no me
hagas repetirlo dos veces el mismo día.

Pipeline **100 % local** de transcripción con diarización (quién habla cuándo)
para notas de voz, con un paso opcional que convierte cada transcripción en
una nota de Obsidian ya redactada. Pensado para el flujo "graba una nota de
voz en el móvil → aparece transcrita, con hablantes separados, en tu vault".

Este pipeline nació para procesar los audios que graba [Aura], una app
Android para tomar notas de voz. No depende de ella ni la necesita: cualquier
carpeta donde vayan apareciendo audios (grabados con el móvil que sea,
copiados a mano, sincronizados como prefieras) sirve como entrada. Si buscas
la app que genera esos audios, está en
[github.com/abel-eiras/Aura](https://github.com/abel-eiras/Aura).

- **Transcripción y diarización: nunca salen de tu máquina.** [WhisperX]
  (sobre [faster-whisper]) para el reconocimiento de voz y el alineamiento por
  palabra; [pyannote] para separar hablantes. Todo corre en local (GPU NVIDIA
  recomendada; funciona en CPU, mucho más despacio).
- **Redacción de notas: opcional y explícita.** Si la activas, cada
  transcripción se manda por texto a un modelo de [OpenRouter] para
  clasificarla y redactar una nota en Markdown. Es lo único que sale de la
  máquina, está desactivado por defecto, y lo activas tú a sabiendas.
- **Idiomas y tipos de nota: configurables**, no hardcodeados. Ver
  [Idioma](#idioma) y [Tipos de nota](#tipos-de-nota).

## Instalación

```bash
git clone <url-de-este-repositorio> aura-transcribe
cd aura-transcribe
./install.sh
```

`install.sh` comprueba las dependencias del sistema, instala las de Python con
[uv], y te pregunta lo mínimo para dejarlo funcionando: dónde aparecen tus
audios, tu idioma (y opcionalmente un segundo idioma), y las claves que
necesites, explicando en cada paso de dónde sacarlas. Se puede volver a
ejecutar cuantas veces haga falta.

### Requisitos

- Linux (usa `flock`, `systemd --user` para la automatización opcional).
- Python 3.12 (lo gestiona `uv`; WhisperX exige `<3.14`).
- `ffmpeg` y `ffprobe` en el PATH.
- GPU NVIDIA con CUDA (opcional: sin ella todo va por CPU).
- Cuenta gratuita de Hugging Face, para el token de la diarización.
- Cuenta de OpenRouter, solo si activas las notas de Obsidian (de pago según uso).

### Instalación manual

Si prefieres no usar el asistente:

```bash
uv sync
cp config.example.toml config.toml   # y edítalo
uv run aura-transcribe check         # diagnóstico: GPU, ffmpeg, token, rutas
```

Las claves van en `~/.config/aura-transcribe/env` (nunca en el repo):

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx
```

#### Token de Hugging Face (diarización)

1. Cuenta gratuita en <https://huggingface.co>.
2. Acepta las condiciones de [pyannote/speaker-diarization-community-1] y de
   [pyannote/segmentation-3.0] (aprobación instantánea).
3. Crea un token de **lectura** en <https://huggingface.co/settings/tokens>.

Sin token se puede transcribir igual con `--no-diarization`.

#### Clave de OpenRouter (solo si quieres notas de Obsidian)

1. Cuenta en <https://openrouter.ai>.
2. Clave en <https://openrouter.ai/keys>.

`[notas] activo = false` por defecto en `config.toml`: actívalo a propósito.

## Qué hace

1. Comprueba que la carpeta de trabajo existe (y, si has configurado un punto
   de montaje, que está montado) y coge un `flock` propio (para que una
   ejecución manual y una programada no se pisen).
2. Busca audios nuevos en la raíz de la carpeta de trabajo (`.wav .ogg .m4a
   .mp3 .opus` por defecto); ignora lo demás y lo que ya tenga JSON en
   `transcription/` o `transcription/sent/`.
3. Descarta ficheros que aún se estén escribiendo/sincronizando (tamaño/mtime
   recientes).
4. Normaliza a WAV 16 kHz mono **en un scratch local**, nunca en la carpeta de trabajo.
5. ASR → alineamiento por palabra → diarización, en serie, liberando VRAM
   entre etapas.
6. Fusiona, limpia y agrega segmentos; escribe el JSON de forma atómica.
7. Solo entonces mueve el audio a `processed/` con `os.rename`.

Nunca se borra nada del usuario. Si algo falla, el audio se queda donde está y
se anota un reintento; tras `[errores] intentos_maximos` pasa a `failed/` con
un `.error.txt` al lado explicando qué pasó.

## Uso

```bash
uv run aura-transcribe run --dry-run          # qué se procesaría (no toca nada)
uv run aura-transcribe run                    # procesa todo el backlog
uv run aura-transcribe run --limit 1          # solo el primero
uv run aura-transcribe run --file nota.wav    # solo ese audio
uv run aura-transcribe run --force            # rehace aunque ya exista el JSON
uv run aura-transcribe run --no-diarization   # sin hablantes (no hace falta token)
uv run aura-transcribe run --language en      # fuerza el idioma
uv run aura-transcribe run --keep-audio       # no mueve el audio a processed/
uv run aura-transcribe check                  # diagnóstico del entorno

uv run aura-transcribe notas                  # notas nuevas (solo transcription/)
uv run aura-transcribe notas --backlog        # también el histórico de transcription/sent/
uv run aura-transcribe notas --force          # regenera aunque ya exista la nota
```

Códigos de salida: `0` ok, `1` error, `2` ya había otra ejecución en marcha, `3` problema de configuración.

Log: `~/.local/state/aura-transcribe/transcribe.log` (rotado, 5 MB × 5).
Estado de reintentos: `~/.local/state/aura-transcribe/estado.json`.

## Configuración

Todo vive en `config.toml` (cópialo de `config.example.toml`, que está
comentado al detalle). Resumen de las secciones:

| Sección | Para qué |
|---|---|
| `[rutas]` | Carpeta de trabajo, subcarpetas, locks, log |
| `[audio]` | Extensiones aceptadas, estabilidad, frecuencia de muestreo |
| `[asr]` | Modelo de WhisperX, dispositivo, tipo de cómputo |
| `[idioma]` | Idioma por defecto y (opcional) un segundo idioma — ver [abajo](#idioma) |
| `[diarizacion]` | Modelo de pyannote, límites de hablantes |
| `[limpieza]` | Fusión y descarte de segmentos |
| `[salida]` | `.txt` opcional, timestamps por palabra |
| `[errores]` | Reintentos antes de mover a `failed/` |
| `[notas]` y `[[notas.categorias]]` | Generación de notas de Obsidian — ver [abajo](#tipos-de-nota) |
| `[ejecucion]` | Aislamiento en subproceso, espera de locks |

Overrides rápidos por variable de entorno: `AURA_CONFIG`, `AURA_BASE_DIR`,
`AURA_PUNTO_MONTAJE`, `AURA_MODELO`, `AURA_DISPOSITIVO`,
`AURA_LOCK_SINCRONIZACION`.

### Idioma

La detección **nunca es libre**, para que un audio corto y ruidoso no acabe
transcrito en un idioma cualquiera:

- Se detecta sobre el primer tramo con **voz** (VAD), no sobre los primeros
  30 s, que pueden ser silencio.
- Solo se contemplan `[idioma] por_defecto` y, si lo configuras, un
  `idioma_alternativo` (pensado para quien trabaja habitualmente en dos
  lenguas, p. ej. una lengua cooficial). Cualquier otra detección se descarta
  y se usa `por_defecto`.
- `confundibles_con_alternativo` son los códigos que el detector suele
  confundir con tu idioma alternativo (dos lenguas próximas se confunden
  entre sí en los primeros segundos); si la probabilidad supera
  `umbral_alternativo`, se transcribe en el idioma alternativo.
- Override manual, por orden de precedencia: `--language xx` > sidecar
  `nombre.lang` > sufijo `_xx` en el nombre del fichero.
- `idiomas_sin_alineamiento` marca los idiomas sin modelo de alineamiento
  forzado disponible: se salta el alineamiento y el hablante se asigna por
  segmento entero en vez de por palabra (queda anotado en
  `metadata.alignment: false`).

Ejemplo real (español + gallego) en `config.example.toml`.

### Tipos de nota

Cada `[[notas.categorias]]` de `config.toml` es un tipo de nota que el
pipeline puede generar. **No hay ninguna categoría especial en el código**:
las tres que trae `config.example.toml` (Reunión / Visita cliente / Nota
personal) son solo un punto de partida pensado para notas de voz de trabajo.
Puedes borrarlas, renombrarlas o añadir las tuyas.

Campos de cada categoría:

| Campo | Obligatorio | Para qué |
|---|---|---|
| `id` | sí | Identificador interno (sin espacios) |
| `etiqueta` | sí | Título que se muestra en la nota |
| `carpeta` | sí | Subcarpeta del vault donde se escribe |
| `tag` | sí | Tag de Obsidian en el frontmatter |
| `criterio` | sí | Una frase: cuándo el LLM debe elegir esta categoría (es lo único que ve el clasificador) |
| `prompt_redaccion` | sí | Prompt de sistema completo para redactar la nota (formato, secciones, idioma si aplica) |
| `modelo` | no | Override de `[notas] modelo` solo para esta categoría (p. ej. un modelo con mejor soporte de un idioma) |

El flujo, por transcripción:

1. Se clasifica con el LLM en uno de los `id` configurados (ve el `full_text`
   completo y el `criterio` de cada categoría; el idioma detectado se le da
   como pista, no como regla).
2. Se redacta con `prompt_redaccion` de esa categoría, usando su `modelo` si
   lo tiene o el general si no.
3. Se escribe el markdown de forma atómica en
   `vault/<carpeta>/<fecha>-<nombre-del-audio>.md`, con frontmatter (`tipo`,
   `fecha`, `audio_origen`, `json_origen`, `idioma`, `hablantes`,
   `duracion_min`, `tags`).

No toca nada de la carpeta de trabajo: solo lee los JSON de `transcription/`
(y, con `--backlog`, `transcription/sent/`) y escribe en el vault local. Si ya
existe una nota para un JSON se omite (por nombre de fichero, sin volver a
clasificar); `--force` la regenera. Un fallo de OpenRouter en un audio no
interrumpe el resto: se registra y se reintenta en la siguiente pasada.

## Salida

Cada audio produce un JSON con este esquema (`metadata` + `segments` +
`full_text`):

```json
{
  "metadata": {
    "source_file": "nota_20260906_161901.wav",
    "processed_at": "2026-09-06T16:22:10Z",
    "duration_seconds": 184.5,
    "language_detected": "es",
    "language_confidence": 0.97,
    "language_used": "es",
    "whisper_model": "large-v3",
    "num_speakers_detected": 2,
    "diarization_model": "pyannote/speaker-diarization-community-1",
    "alignment": true,
    "pipeline_version": "0.1.0",
    "processing_seconds": 42.1
  },
  "segments": [
    {"id": 0, "speaker": "SPEAKER_00", "start": 0.32, "end": 4.1, "text": "..."}
  ],
  "full_text": "SPEAKER_00: ...\nSPEAKER_01: ..."
}
```

Opcionalmente (`[salida]` en `config.toml`) un `.txt` legible al lado y
`words` con timestamps por palabra dentro de cada segmento.

## Automatización (`systemd --user`)

```bash
cd systemd
./instalar.sh              # enlaza las unidades, genera el envoltorio y arranca el timer
./instalar.sh --copiar     # igual, pero copiando las unidades en vez de enlazarlas
./instalar.sh --desinstalar
```

Se ejecuta cada 15 minutos. Ver los comentarios de `systemd/aura-transcribe.service`
y `systemd/aura-transcribe.timer` para los porqués (timeout infinito porque
`Type=oneshot` mataría una transcripción larga, cadencia pensada para no
competir con un sincronizador externo si tienes uno, etc.).

```bash
systemctl --user list-timers aura-transcribe.timer      # cuándo toca la próxima
systemctl --user status aura-transcribe.service         # cómo fue la última
journalctl --user -u aura-transcribe.service -f         # en vivo
tail -f ~/.local/state/aura-transcribe/transcribe.log   # log propio del pipeline
```

## Coordinación con un sincronizador externo (opcional)

Si la carpeta de trabajo la sincroniza también otro proceso tuyo (por ejemplo
un `rclone bisync` con Google Drive o Nextcloud, protegido con su propio
`flock`), configura `[rutas] lock_sincronizacion` apuntando a ese mismo
fichero de lock. El pipeline lo cogerá, pero **solo durante la ventana de
mutación** (escribir el JSON y mover el audio: milisegundos), nunca durante
toda la transcripción. Desactivado por defecto (`lock_sincronizacion` vacío):
si tu carpeta de trabajo es simplemente local, no necesitas tocar esto.

## Arquitectura, en dos decisiones no obvias

- **Aislamiento en subproceso.** Cada audio se procesa en su propio
  subproceso, que muere al terminar. En una GPU con poca VRAM, la
  fragmentación se acumula a lo largo de un proceso de vida larga y puede
  reventar un lote de varios audios con `CUDA out of memory` aunque se libere
  caché entre etapas; la única forma fiable de devolver TODA la VRAM es que el
  proceso muera. Se puede desactivar (`[ejecucion] aislar_subproceso = false`
  o `--sin-aislamiento`) si tu GPU tiene VRAM de sobra: algo más rápido por
  lote, a cambio de ese riesgo.
- **Detección de idioma restringida.** Nunca "libre": solo se contemplan el
  idioma por defecto y, si lo configuras, uno alternativo. Cualquier otra
  detección (un audio corto y ruidoso "detectado" como italiano, por ejemplo)
  se descarta y se usa el idioma por defecto. Ver [Idioma](#idioma).

Detalles adicionales (contrato del subproceso, por qué pyannote recibe el
audio ya cargado en memoria, etc.) están comentados donde corresponde en el
código: `pipeline.py`, `worker.py`, `diarize.py`.

## Tests

```bash
uv run pytest tests -q
```

No tocan ni la GPU ni ninguna carpeta real: trabajan sobre árboles temporales
y mockean las llamadas a OpenRouter.

## Privacidad, en resumen

- Transcripción y diarización: **100 % local**, nada sale de tu máquina.
- Notas de Obsidian: **opcional y desactivada por defecto**; si la activas,
  el texto completo de cada transcripción se manda a OpenRouter (y de ahí al
  proveedor del modelo que elijas) para clasificarla y redactarla.
- Las claves (`HF_TOKEN`, `OPENROUTER_API_KEY`) viven en
  `~/.config/aura-transcribe/env`, fuera del repositorio, con permisos `600`.
  Nunca se registran en el log ni se escriben en ningún fichero versionado.

## Licencia

[MIT](LICENSE).

## Contribuir

Ver [CONTRIBUTING.md](CONTRIBUTING.md).

[Aura]: https://github.com/abel-eiras/Aura
[WhisperX]: https://github.com/m-bain/whisperX
[faster-whisper]: https://github.com/SYSTRAN/faster-whisper
[pyannote]: https://github.com/pyannote/pyannote-audio
[OpenRouter]: https://openrouter.ai
[uv]: https://docs.astral.sh/uv/
[pyannote/speaker-diarization-community-1]: https://huggingface.co/pyannote/speaker-diarization-community-1
[pyannote/segmentation-3.0]: https://huggingface.co/pyannote/segmentation-3.0
