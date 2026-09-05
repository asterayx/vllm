#!/usr/bin/env bash
# Install cloudflared as a systemd service (not a container).
# `sudo cloudflared service install` looks in /root/.cloudflared and
# /etc/cloudflared — not the invoking user's ~/.cloudflared.
#
#   cloudflared tunnel login
#   cloudflared tunnel create spark-obs
#   # write ~/.cloudflared/config.yml (or set TUNNEL_ID + HOSTNAME)
#   ./install-service.sh
set -euo pipefail

SRC="${CLOUDFLARED_SRC:-${HOME}/.cloudflared}"
DEST="${CLOUDFLARED_DEST:-/etc/cloudflared}"
HOSTNAME="${CLOUDFLARED_HOSTNAME:-token.asteraix.com}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not on PATH. curl the arm64 .deb, do not use a container." >&2
  exit 1
fi
if [[ ! -d "${SRC}" ]]; then
  echo "missing ${SRC}; run cloudflared tunnel login / create first" >&2
  exit 1
fi

if [[ -z "${TUNNEL_ID:-}" ]]; then
  shopt -s nullglob
  creds=("${SRC}"/*.json)
  shopt -u nullglob
  if [[ ${#creds[@]} -eq 1 ]]; then
    TUNNEL_ID="$(basename "${creds[0]}" .json)"
  elif [[ -f "${SRC}/config.yml" ]] && grep -qE '^tunnel:' "${SRC}/config.yml"; then
    TUNNEL_ID="$(awk '/^tunnel:/{print $2; exit}' "${SRC}/config.yml")"
  fi
fi
if [[ -z "${TUNNEL_ID:-}" ]]; then
  echo "set TUNNEL_ID (cloudflared tunnel list)" >&2
  ls -l "${SRC}" >&2 || true
  exit 1
fi
if [[ ! -f "${SRC}/${TUNNEL_ID}.json" ]]; then
  echo "missing credentials ${SRC}/${TUNNEL_ID}.json" >&2
  exit 1
fi

echo "installing tunnel ${TUNNEL_ID} -> ${DEST} (hostname ${HOSTNAME})"
sudo mkdir -p "${DEST}"
sudo cp -f "${SRC}/${TUNNEL_ID}.json" "${DEST}/${TUNNEL_ID}.json"
if [[ -f "${SRC}/cert.pem" ]]; then
  sudo cp -f "${SRC}/cert.pem" "${DEST}/cert.pem"
fi

if [[ -f "${SRC}/config.yml" ]]; then
  sudo sed \
    -e "s|^tunnel:.*|tunnel: ${TUNNEL_ID}|" \
    -e "s|^credentials-file:.*|credentials-file: ${DEST}/${TUNNEL_ID}.json|" \
    "${SRC}/config.yml" | sudo tee "${DEST}/config.yml" >/dev/null
else
  sudo tee "${DEST}/config.yml" >/dev/null <<EOF
tunnel: ${TUNNEL_ID}
credentials-file: ${DEST}/${TUNNEL_ID}.json

ingress:
  - hostname: ${HOSTNAME}
    path: /dash(/.*)?
    service: http://127.0.0.1:3000
    originRequest:
      httpHostHeader: ${HOSTNAME}
      disableChunkedEncoding: true

  - hostname: ${HOSTNAME}
    path: /configs(/.*)?
    service: http://127.0.0.1:30000
    originRequest:
      httpHostHeader: ${HOSTNAME}
      disableChunkedEncoding: true

  - hostname: ${HOSTNAME}
    path: /healthz
    service: http://127.0.0.1:30000

  - service: http_status:404
EOF
fi

if systemctl list-unit-files cloudflared.service >/dev/null 2>&1 \
    && [[ -f /etc/systemd/system/cloudflared.service ]]; then
  echo "cloudflared.service already installed; not running service install"
else
  sudo cloudflared --config "${DEST}/config.yml" service install
fi
sudo systemctl daemon-reload
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared --no-pager
echo "login:     https://${HOSTNAME}/dash"
echo "configs:   https://${HOSTNAME}/configs"
