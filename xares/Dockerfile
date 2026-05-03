# gpu-summon Full-Stack Image
#
# Combines on a single vast.ai-rentable container:
#   - llama.cpp/llama-server (CUDA) for inference
#   - dockerd (DinD) so xaresaicoder's docker-compose stack can run inside
#   - Caddy with the duckdns DNS-01 plugin for Wildcard-TLS
#   - git/curl/envsubst for the bootstrap onstart.sh
#
# Built and pushed by the maintainer via xares/build-and-push.sh.
# vast.ai instances must be created with --privileged for dockerd to work.

ARG LLAMACPP_VERSION=server-cuda
FROM ghcr.io/ggml-org/llama.cpp:${LLAMACPP_VERSION} AS llamacpp

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps + Docker (we ship dockerd, the daemon starts in onstart.sh).
#
# vast.ai's /.launch wrapper runs BEFORE our onstart_cmd and tries to apt-install
# openssh-server / tmux / rsync / software-properties-common at boot time. If
# the rented host can't reach archive.ubuntu.com (which we have observed in the
# wild) those installs fail and /.launch loops forever on `ssh: command not
# found`. We pre-install the lot so /.launch's apt calls become no-ops and the
# image works regardless of upstream-mirror reachability.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git gettext-base \
        iptables uidmap fuse-overlayfs \
        netcat-openbsd jq \
        openssh-server openssh-client openssh-sftp-server \
        rsync tmux software-properties-common \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
        https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Caddy with the duckdns DNS-01 plugin baked in. We use the official xcaddy
# build container to produce a static binary, then copy it into our image.
RUN curl -fsSL https://caddyserver.com/api/download?os=linux\&arch=amd64\&p=github.com%2Fcaddy-dns%2Fduckdns \
        -o /usr/local/bin/caddy \
    && chmod +x /usr/local/bin/caddy \
    && /usr/local/bin/caddy version

# llama-server binary + libs from the upstream image. Keep the same paths
# the existing summon.py manual-restart docs assume (/app + LD_LIBRARY_PATH=/app).
COPY --from=llamacpp /app /app

ENV LD_LIBRARY_PATH=/app
ENV HF_HOME=/workspace/hf_cache

# Bootstrap files. The actual templates live in /opt; onstart.sh renders
# them via envsubst at container start.
COPY onstart.sh           /opt/onstart.sh
COPY Caddyfile.template   /opt/Caddyfile.template
COPY xares-env.template   /opt/xares-env.template
RUN chmod +x /opt/onstart.sh

EXPOSE 22 80 443 8080

# vast.ai sets `onstart_cmd` per instance. To keep this image usable both via
# vast (where the entry comes from create_instance(onstart_cmd=...)) and via a
# plain `docker run` for local sanity checks, we default-CMD onstart.sh.
CMD ["/opt/onstart.sh"]
