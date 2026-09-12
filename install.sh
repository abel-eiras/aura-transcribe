#!/usr/bin/env bash
# Asistente de instalación de aura-transcribe.
#
# Comprueba las dependencias del sistema, instala las de Python con uv, y
# pregunta lo mínimo para dejar config.toml y las claves listas:
#   - carpeta donde aparecen los audios nuevos,
#   - idioma principal y (opcional) un segundo idioma,
#   - token de Hugging Face (diarización),
#   - si quieres notas de Obsidian vía OpenRouter, y su clave.
#
# No hace nada irreversible sin preguntar. Se puede volver a ejecutar tantas
# veces como haga falta: vuelve a preguntar todo, pero solo sobrescribe lo que
# confirmes.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${HOME}/.config/aura-transcribe"
ENV_FILE="${ENV_DIR}/env"
CONFIG_FILE="${REPO_DIR}/config.toml"
EJEMPLO="${REPO_DIR}/config.example.toml"

verde() { printf '\033[32m%s\033[0m\n' "$1"; }
amarillo() { printf '\033[33m%s\033[0m\n' "$1"; }
rojo() { printf '\033[31m%s\033[0m\n' "$1"; }

preguntar() {  # preguntar "texto" "valor_por_defecto" -> imprime la respuesta
  local texto="$1" defecto="${2:-}" respuesta
  if [[ -n "$defecto" ]]; then
    read -r -p "$texto [$defecto]: " respuesta || true
    echo "${respuesta:-$defecto}"
  else
    read -r -p "$texto: " respuesta || true
    echo "$respuesta"
  fi
}

confirmar() {  # confirmar "texto" [s/N por defecto] -> 0 si sí, 1 si no
  local texto="$1" defecto="${2:-n}" respuesta
  local sufijo="s/N"
  [[ "$defecto" == "s" ]] && sufijo="S/n"
  read -r -p "$texto [$sufijo]: " respuesta || true
  respuesta="${respuesta:-$defecto}"
  [[ "$respuesta" =~ ^[sSyY] ]]
}

echo
verde "== Instalación de aura-transcribe =="
echo "Repositorio: $REPO_DIR"
echo

# --- 1. Dependencias del sistema -------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
  rojo "Falta python3. Instálalo con el gestor de paquetes de tu distro y vuelve a lanzar este script."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  rojo "Falta 'uv' (gestor de paquetes de Python)."
  echo "Instálalo con:"
  echo
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo
  echo "y vuelve a ejecutar ./install.sh."
  exit 1
fi

FALTAN_BIN=()
for binario in ffmpeg ffprobe; do
  command -v "$binario" >/dev/null 2>&1 || FALTAN_BIN+=("$binario")
done
if [[ ${#FALTAN_BIN[@]} -gt 0 ]]; then
  amarillo "Faltan en el PATH: ${FALTAN_BIN[*]}"
  echo "Instálalos con el gestor de paquetes de tu distro, por ejemplo:"
  echo "  Fedora/Nobara: sudo dnf install ffmpeg"
  echo "  Debian/Ubuntu: sudo apt install ffmpeg"
  echo "  Arch:          sudo pacman -S ffmpeg"
  confirmar "¿Continuar de todas formas?" n || exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  verde "GPU NVIDIA detectada. Recuerda tener instalado el driver CUDA correspondiente."
else
  amarillo "No se detectó una GPU NVIDIA (nvidia-smi no está en el PATH)."
  echo "El pipeline funciona igual en CPU, pero mucho más despacio."
fi

echo
verde "Instalando dependencias de Python con uv (puede tardar varios minutos:"
echo "descarga PyTorch y los modelos de WhisperX/pyannote)..."
(cd "$REPO_DIR" && uv sync)

# --- 2. config.toml ----------------------------------------------------

GENERAR_CONFIG=1
if [[ -f "$CONFIG_FILE" ]]; then
  amarillo "Ya existe config.toml."
  if confirmar "¿Regenerarlo desde cero? (perderás los cambios manuales que le hayas hecho)" n; then
    GENERAR_CONFIG=1
  else
    GENERAR_CONFIG=0
  fi
fi

if [[ "$GENERAR_CONFIG" -eq 1 ]]; then
  echo
  echo "-- Carpeta de audios --"
  echo "Es la carpeta donde tu(s) dispositivo(s) dejan las notas de voz nuevas"
  echo "(por ejemplo, sincronizada por Syncthing, una app de Drive, o copiando a mano)."
  BASE_DIR="$(preguntar "Ruta de la carpeta de audios" "$HOME/AudioNotes/aura")"
  BASE_DIR="${BASE_DIR/#\~/$HOME}"
  mkdir -p "$BASE_DIR"

  echo
  echo "-- Idioma --"
  IDIOMA_DEFECTO="$(preguntar "Código de idioma principal (ISO 639-1: es, en, fr, de...)" "es")"

  IDIOMA_ALT=""
  CONFUNDIBLES=""
  UMBRAL_ALT="0.5"
  if confirmar "¿Detectas también un segundo idioma en tus audios (p. ej. una lengua cooficial)?" n; then
    IDIOMA_ALT="$(preguntar "Código del segundo idioma" "")"
    CONFUNDIBLES="$(preguntar "Códigos que el detector podría confundir con él, separados por comas" "$IDIOMA_ALT")"
    UMBRAL_ALT="$(preguntar "Probabilidad mínima para aceptarlo (0-1)" "0.5")"
  fi

  echo
  python3 - "$EJEMPLO" "$CONFIG_FILE" "$BASE_DIR" "$IDIOMA_DEFECTO" "$IDIOMA_ALT" "$CONFUNDIBLES" "$UMBRAL_ALT" <<'PY'
import sys

ejemplo, destino, base_dir, idioma_defecto, idioma_alt, confundibles, umbral_alt = sys.argv[1:8]

texto = open(ejemplo, encoding="utf-8").read()

def toml_str(valor: str) -> str:
    return '"' + valor.replace("\\", "\\\\").replace('"', '\\"') + '"'

texto = texto.replace(
    'base = "~/AudioNotes/aura"',
    f"base = {toml_str(base_dir)}",
    1,
)
texto = texto.replace(
    'por_defecto = "es"',
    f"por_defecto = {toml_str(idioma_defecto)}",
    1,
)

if idioma_alt.strip():
    lista_confundibles = [c.strip() for c in confundibles.split(",") if c.strip()] or [idioma_alt.strip()]
    permitidos_toml = "[" + ", ".join(toml_str(x) for x in (idioma_defecto, idioma_alt)) + "]"
    confundibles_toml = "[" + ", ".join(toml_str(x) for x in lista_confundibles) + "]"
    texto = texto.replace('permitidos = ["es"]', f"permitidos = {permitidos_toml}", 1)
    texto = texto.replace(
        '# idioma_alternativo = "gl"',
        f"idioma_alternativo = {toml_str(idioma_alt)}",
        1,
    )
    texto = texto.replace(
        '# confundibles_con_alternativo = ["gl", "pt"]',
        f"confundibles_con_alternativo = {confundibles_toml}",
        1,
    )
    texto = texto.replace(
        "umbral_alternativo = 0.5",
        f"umbral_alternativo = {float(umbral_alt)}",
        1,
    )
    # Si el segundo idioma no tiene modelo de alineamiento forzado disponible,
    # añade su código a `idiomas_sin_alineamiento` a mano en config.toml.
else:
    permitidos_toml = "[" + toml_str(idioma_defecto) + "]"
    texto = texto.replace('permitidos = ["es"]', f"permitidos = {permitidos_toml}", 1)

with open(destino, "w", encoding="utf-8") as fh:
    fh.write(texto)

print(f"Escrito {destino}")
PY

  verde "config.toml generado."
  echo "Los tipos de nota (Reunión / Visita cliente / Nota personal) y sus"
  echo "prompts se han copiado tal cual de config.example.toml: edítalos, quítalos"
  echo "o añade los tuyos en la sección [[notas.categorias]] cuando quieras."
fi

# --- 3. Claves --------------------------------------------------------

mkdir -p "$ENV_DIR"
chmod 700 "$ENV_DIR"
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"

guardar_clave() {  # guardar_clave NOMBRE valor
  local nombre="$1" valor="$2"
  if grep -q "^${nombre}=" "$ENV_FILE" 2>/dev/null; then
    sed -i "s#^${nombre}=.*#${nombre}=${valor}#" "$ENV_FILE"
  else
    printf '%s=%s\n' "$nombre" "$valor" >> "$ENV_FILE"
  fi
}

echo
verde "-- Token de Hugging Face (necesario para la diarización) --"
if grep -q "^HF_TOKEN=" "$ENV_FILE" 2>/dev/null; then
  amarillo "Ya hay un HF_TOKEN guardado en $ENV_FILE."
  confirmar "¿Sustituirlo?" n && ACTUALIZAR_HF=1 || ACTUALIZAR_HF=0
else
  ACTUALIZAR_HF=1
fi

if [[ "$ACTUALIZAR_HF" -eq 1 ]]; then
  echo "Pasos (una sola vez, es gratis):"
  echo "  1. Crea una cuenta en https://huggingface.co"
  echo "  2. Acepta las condiciones de estos dos modelos (aprobación instantánea):"
  echo "       https://huggingface.co/pyannote/speaker-diarization-community-1"
  echo "       https://huggingface.co/pyannote/segmentation-3.0"
  echo "  3. Crea un token de LECTURA en https://huggingface.co/settings/tokens"
  echo
  HF_TOKEN_VALOR="$(preguntar "Pega aquí el token (o deja en blanco para configurarlo más tarde)" "")"
  if [[ -n "$HF_TOKEN_VALOR" ]]; then
    guardar_clave "HF_TOKEN" "$HF_TOKEN_VALOR"
    verde "Guardado en $ENV_FILE."
  else
    amarillo "Sin token: podrás transcribir con --no-diarization mientras tanto."
  fi
fi

echo
verde "-- Notas de Obsidian (opcional, vía OpenRouter) --"
echo "Esta función manda el TEXTO COMPLETO de cada transcripción a OpenRouter"
echo "para clasificarla y redactar una nota. Es opcional y está desactivada"
echo "por defecto: actívala solo si te parece bien que ese texto salga de tu"
echo "máquina."
if confirmar "¿Activar la generación de notas de Obsidian?" n; then
  echo "Pasos (una sola vez):"
  echo "  1. Crea una cuenta en https://openrouter.ai"
  echo "  2. Genera una clave en https://openrouter.ai/keys"
  echo "     (OpenRouter cobra por uso; revisa los precios del modelo que elijas)"
  echo
  OPENROUTER_VALOR="$(preguntar "Pega aquí la clave (o deja en blanco para configurarla más tarde)" "")"
  if [[ -n "$OPENROUTER_VALOR" ]]; then
    guardar_clave "OPENROUTER_API_KEY" "$OPENROUTER_VALOR"
    verde "Guardado en $ENV_FILE."
  else
    amarillo "Sin clave todavía: añádela luego en $ENV_FILE como OPENROUTER_API_KEY=sk-or-..."
  fi
  if [[ -f "$CONFIG_FILE" ]]; then
    python3 - "$CONFIG_FILE" <<'PY'
import sys
ruta = sys.argv[1]
texto = open(ruta, encoding="utf-8").read()
texto = texto.replace("activo = false", "activo = true", 1)
open(ruta, "w", encoding="utf-8").write(texto)
PY
    verde "[notas] activo = true en config.toml."
  else
    amarillo "No hay config.toml todavía: pon [notas] activo = true a mano cuando lo crees."
  fi
else
  echo "Puedes activarlo más tarde poniendo [notas] activo = true en config.toml."
fi

# --- 4. Resumen ---------------------------------------------------------

echo
verde "== Listo =="
echo "config.toml:  $CONFIG_FILE"
echo "claves:       $ENV_FILE (permisos 600, nunca se versiona)"
echo
echo "Siguientes pasos:"
echo "  cd \"$REPO_DIR\""
echo "  uv run aura-transcribe check       # diagnóstico del entorno"
echo "  uv run aura-transcribe run --dry-run"
echo
echo "Para que se ejecute solo cada 15 minutos (systemd --user), ver:"
echo "  systemd/instalar.sh"
echo
echo "Para cambiar los tipos de nota, sus prompts, o afinar cualquier otro"
echo "parámetro, edita config.toml directamente (está comentado)."
