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
# /workspace/projects gehoert dem coder-User (code-server laeuft als coder).
chown -R coder:coder /workspace/projects 2>/dev/null || true

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
#    default project directory the user lands in. Runs as `coder` so the
#    browser terminal isn't root (sudo without password if needed).
log "starting code-server as coder (PASSWORD=set, bind 127.0.0.1:8443)"
# HOME explicit: 'sudo -H' is overridden by sudoers' env_keep += HOME on the
# upstream image, so HOME stays /root and code-server tries to mkdir
# /root/.config/code-server (EACCES, instant crash). Setting HOME via env
# bypasses sudo's env_keep policy and points code-server at /home/coder.
#
# --proxy-domain tells code-server its public hostname so it knows when to
# enable proxy-mode (forwarded ports get pretty subdomain URLs in the Ports
# tab instead of the path-based /proxy/<port>/). VSCODE_PROXY_URI is the
# template VS Code uses to render those URLs. Both need the host port that
# vast assigned (random, e.g. :44145) -- but onstart.sh doesn't know that
# port from inside the container. summon.py SSHes in post-launch and runs
# /opt/restart-codeserver.sh <host_port> with the real port.
sudo -u coder env HOME=/home/coder PASSWORD="$CODESERVER_PASSWORD" \
    setsid nohup code-server \
    --bind-addr 127.0.0.1:8443 \
    --auth password \
    --disable-telemetry \
    --proxy-domain "$DOMAIN" \
    /workspace/projects \
    > /var/log/code-server.log 2>&1 < /dev/null &
disown || true

# Helper invoked by summon.py over SSH once the vast host port is known.
# Restarts code-server with VSCODE_PROXY_URI baked at the right :PORT so
# the Ports tab shows usable URLs.
cat > /opt/restart-codeserver.sh <<'EOSH'
#!/bin/bash
# usage: /opt/restart-codeserver.sh <host_port_for_443>
set -e
HOST_PORT="${1:?missing host port arg}"
# Read DOMAIN/CODESERVER_PASSWORD from PID 1 env (where vast set them).
DOMAIN=$(grep -z '^DOMAIN=' /proc/1/environ | tr -d '\0' | cut -d= -f2-)
PASSWORD=$(grep -z '^CODESERVER_PASSWORD=' /proc/1/environ | tr -d '\0' | cut -d= -f2-)
[[ -n "$DOMAIN" && -n "$PASSWORD" ]] || { echo "missing env"; exit 1; }
URI="https://{{port}}.${DOMAIN}:${HOST_PORT}/"
pkill -f /usr/lib/code-server 2>/dev/null || true
sleep 1
sudo -u coder env HOME=/home/coder PASSWORD="$PASSWORD" VSCODE_PROXY_URI="$URI" \
    setsid nohup code-server \
    --bind-addr 127.0.0.1:8443 \
    --auth password --disable-telemetry \
    --proxy-domain "$DOMAIN" \
    /workspace/projects > /var/log/code-server.log 2>&1 < /dev/null &
disown || true
echo "code-server restarted with VSCODE_PROXY_URI=$URI"
EOSH
chmod +x /opt/restart-codeserver.sh

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

# 4) Render AGENTS.md so any AI coding agent that opens /workspace/projects
#    sees the dev environment. The host-port suffix isn't visible from
#    inside the container -- summon.py SSHes in after readiness and patches
#    the placeholder via sed (or leaves it for the user to replace).
log "rendering /workspace/projects/AGENTS.md"
DOMAIN="$DOMAIN" LLAMA_MODEL="$LLAMA_MODEL" \
    envsubst '${DOMAIN} ${LLAMA_MODEL}' \
    < /opt/AGENTS.md.template > /workspace/projects/AGENTS.md
chown coder:coder /workspace/projects/AGENTS.md

# 5) Write opencode config for in-container use AS SOON AS llama-server is
#    ready. Background loop so we don't block onstart on the 5-15min model
#    download. Once written, the user can open a terminal in code-server
#    and just type `opencode`. baseURL points at localhost (no Caddy
#    detour for the local case), uses the real model-id the server reports
#    (which differs from the raw HF spec like 'unsloth/...').
(
    LLM_URL="http://127.0.0.1:8080/v1/models"
    AUTH_HDR="Authorization: Bearer $LLAMA_API_KEY"

    log "[opencode-cfg] waiting for llama-server to report a model..."
    while true; do
        if curl -sf -H "$AUTH_HDR" "$LLM_URL" > /tmp/models.json 2>/dev/null; then
            MODEL_ID=$(jq -r '.data[0].id // empty' /tmp/models.json 2>/dev/null)
            if [[ -n "$MODEL_ID" ]]; then
                break
            fi
        fi
        sleep 30
    done
    rm -f /tmp/models.json

    log "[opencode-cfg] writing config (model=$MODEL_ID)"
    # Config-Dateien fuer den coder-User schreiben (code-server-Terminal
    # landet als coder), Ownership korrigieren.
    mkdir -p /home/coder/.config/opencode /home/coder/.local/share/opencode

    cat > /home/coder/.config/opencode/opencode.json <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "local-llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local llama-server",
      "options": {"baseURL": "http://localhost:8080/v1"},
      "models": {"$MODEL_ID": {"tools": true}}
    }
  },
  "model": "local-llamacpp/$MODEL_ID",
  "small_model": "local-llamacpp/$MODEL_ID"
}
JSON

    cat > /home/coder/.local/share/opencode/auth.json <<JSON
{
  "local-llamacpp": {"type": "api", "key": "$LLAMA_API_KEY"}
}
JSON

    chown -R coder:coder /home/coder/.config /home/coder/.local

    log "[opencode-cfg] done. type 'opencode' in code-server terminal."
) &
disown || true

log "all services launched. log files for SSH-side debugging:"
log "  - /var/log/llama-server.log    (model download + serving)"
log "  - /var/log/code-server.log     (IDE startup)"
log "  - /var/log/caddy.log           (TLS / ACME / routing)"
log "  - /var/log/onstart.log         (this script's own output)"

# Block forever. Without this, the container exits as soon as we return.
sleep infinity
