#!/bin/bash
# gpu-summon code-server onstart
#
# Runs inside an unprivileged vast.ai container. Brings up three native
# host processes (no Docker, since vast doesn't allow --privileged):
#   1) llama-server (uses GPU directly)
#   2) code-server (browser-based VS Code, password-protected)
#   3) Caddy (TLS termination + routing via duckdns DNS-01 wildcard cert)
#
# Required env (injected by summon.py):
#   DOMAIN                 e.g. mybox.duckdns.org
#   DUCKDNS_TOKEN          duckdns API token (for ACME DNS-01)
#   LLAMA_API_KEY          Bearer token for llama-server
#   CODESERVER_PASSWORD    workspace password for code-server's login page
# Optional:
#   LLAMA_MODEL            default: unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL
#   LLAMA_CTX              default: 65536
#   LLAMA_PARALLEL         default: 4
#   ACME_EMAIL             default: admin@${DOMAIN}

set -euo pipefail

LLAMA_MODEL="${LLAMA_MODEL:-unsloth/Qwen3.6-27B-GGUF:UD-Q5_K_XL}"
LLAMA_CTX="${LLAMA_CTX:-65536}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-4}"
ACME_EMAIL="${ACME_EMAIL:-admin@${DOMAIN}}"

log() { echo "[onstart $(date -u +%H:%M:%S)] $*"; }

require() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        log "FATAL: required env var $name is empty"
        exit 1
    fi
}

require DOMAIN
require DUCKDNS_TOKEN
require LLAMA_API_KEY
require CODESERVER_PASSWORD

mkdir -p /var/log /etc/caddy /workspace/hf_cache /workspace/projects

# 1) llama-server. Same flags as the standalone llamacpp backend.
#    --host 127.0.0.1 because Caddy fronts it on the public side.
log "starting llama-server (model=$LLAMA_MODEL ctx=$LLAMA_CTX parallel=$LLAMA_PARALLEL)"
HF_HOME=/workspace/hf_cache LD_LIBRARY_PATH=/app setsid nohup /app/llama-server \
    -hf "$LLAMA_MODEL" \
    --host 127.0.0.1 --port 8080 \
    --n-gpu-layers 999 \
    --ctx-size "$LLAMA_CTX" \
    --parallel "$LLAMA_PARALLEL" \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --flash-attn on --jinja \
    --chat-template-kwargs '{"preserve_thinking":true}' \
    --temp 0.7 --top-p 0.8 --top-k 20 \
    --presence-penalty 1.5 --min-p 0.0 \
    --api-key "$LLAMA_API_KEY" \
    --metrics \
    > /var/log/llama-server.log 2>&1 < /dev/null &
disown || true

# 2) code-server. PASSWORD env enables the built-in login form. Bind to
#    127.0.0.1 so only Caddy can reach it. /workspace/projects is the
#    default project directory the user lands in.
log "starting code-server (PASSWORD=set, bind 127.0.0.1:8443)"
PASSWORD="$CODESERVER_PASSWORD" setsid nohup code-server \
    --bind-addr 127.0.0.1:8443 \
    --auth password \
    --disable-telemetry \
    /workspace/projects \
    > /var/log/code-server.log 2>&1 < /dev/null &
disown || true

# 3) Caddy. Wildcard cert via duckdns DNS-01. Caddy retries cert acquisition
#    on its own; if duckdns/LE hiccup, watch /var/log/caddy.log via SSH.
log "rendering Caddyfile"
DOMAIN="$DOMAIN" \
DUCKDNS_TOKEN="$DUCKDNS_TOKEN" \
ACME_EMAIL="$ACME_EMAIL" \
    envsubst '${DOMAIN} ${DUCKDNS_TOKEN} ${ACME_EMAIL}' \
    < /opt/Caddyfile.template > /etc/caddy/Caddyfile

log "starting caddy"
caddy run --config /etc/caddy/Caddyfile --adapter caddyfile \
    > /var/log/caddy.log 2>&1 &
disown || true

log "all services launched. log files for SSH-side debugging:"
log "  - /var/log/llama-server.log    (model download + serving)"
log "  - /var/log/code-server.log     (IDE startup)"
log "  - /var/log/caddy.log           (TLS / ACME / routing)"

# Block forever. Without this, the container exits as soon as we return.
sleep infinity
