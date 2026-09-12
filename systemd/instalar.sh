#!/usr/bin/env bash
# Instala (o desinstala) la automatización de aura-transcribe en systemd --user.
#
# Las unidades (.service, .timer) viven en el repositorio, aquí al lado, y no
# llevan ninguna ruta personal: se enlazan (o copian) tal cual a
# ~/.config/systemd/user. El envoltorio (aura-transcribe.sh) SÍ necesita saber
# dónde está este repo y dónde está `uv`, así que se genera a partir de
# aura-transcribe.sh.template sustituyendo esas dos rutas, y se escribe
# siempre de cero en ~/.local/bin (nunca se enlaza).
#
# Uso:
#   ./instalar.sh              instala y arranca el timer
#   ./instalar.sh --copiar     igual, pero copiando las unidades en vez de enlazarlas
#   ./instalar.sh --desinstalar  para el timer y quita lo instalado

set -euo pipefail

ORIGEN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROYECTO="$(cd "$ORIGEN/.." && pwd)"
DESTINO_UNIDADES="${HOME}/.config/systemd/user"
DESTINO_BIN="${HOME}/.local/bin"
UNIDADES=(aura-transcribe.service aura-transcribe.timer)
SCRIPT=aura-transcribe.sh

modo=enlazar
case "${1:-}" in
  --copiar) modo=copiar ;;
  --desinstalar) modo=desinstalar ;;
  "") ;;
  *) echo "Opción desconocida: $1" >&2; exit 2 ;;
esac

mkdir -p "$DESTINO_UNIDADES" "$DESTINO_BIN"

if [[ "$modo" == "desinstalar" ]]; then
  systemctl --user disable --now aura-transcribe.timer 2>/dev/null || true
  for unidad in "${UNIDADES[@]}"; do
    rm -f "${DESTINO_UNIDADES}/${unidad}"
  done
  rm -f "${DESTINO_BIN}/${SCRIPT}"
  systemctl --user daemon-reload
  echo "Desinstalado. Ningún sincronizador externo se ha tocado."
  exit 0
fi

UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
  UV="${HOME}/.local/bin/uv"
fi
if [[ ! -x "$UV" ]]; then
  echo "No encuentro 'uv' (ni en el PATH ni en $UV). Instálalo antes de continuar." >&2
  exit 1
fi

echo "Proyecto: $PROYECTO"
echo "uv:       $UV"

# El envoltorio SIEMPRE se regenera desde la plantilla: lleva rutas absolutas
# resueltas ahora mismo, así que enlazarlo (en vez de generarlo) rompería
# --copiar y cualquier repo movido de sitio.
sed -e "s#__AURA_PROYECTO__#${PROYECTO}#g" -e "s#__AURA_UV__#${UV}#g" \
  "${ORIGEN}/${SCRIPT}.template" > "${DESTINO_BIN}/${SCRIPT}"
chmod 0755 "${DESTINO_BIN}/${SCRIPT}"

instalar_unidad() {  # $1 = fichero
  if [[ "$modo" == "copiar" ]]; then
    install -m 0644 "${ORIGEN}/${1}" "${DESTINO_UNIDADES}/${1}"
  else
    ln -sfn "${ORIGEN}/${1}" "${DESTINO_UNIDADES}/${1}"
  fi
}

for unidad in "${UNIDADES[@]}"; do
  instalar_unidad "$unidad"
done

systemctl --user daemon-reload
systemctl --user enable --now aura-transcribe.timer

echo
echo "Instalado. Estado del timer:"
systemctl --user list-timers aura-transcribe.timer --no-pager

echo
echo "Nota: aura-transcribe.service tiene 'After=sync-aura-drive.service' por si"
echo "tienes un sincronizador externo con ese nombre de unidad; si no lo tienes,"
echo "no pasa nada (la línea se ignora), pero puedes borrarla del .service si"
echo "prefieres no verla."
