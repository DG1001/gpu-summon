# gpu-summon code-server image
#
# Combines on a single vast.ai-rentable container:
#   - llama.cpp/llama-server (CUDA) for inference
#   - code-server (browser-based VS Code)
#   - Caddy with the duckdns DNS-01 plugin for Wildcard-TLS
#
# No Docker-in-Docker: vast.ai standard offers do NOT allow --privileged,
# so dockerd cannot configure iptables and dies at boot. Everything runs
# as native host processes here. See LESSONS.md for the painful detour.
#
# Built and pushed by the maintainer via codeserver/build-and-push.sh
# (or the build-codeserver-image.yml GitHub Actions workflow).

# Use the upstream llama.cpp image as the base directly. Its base distro
# (currently Ubuntu 24.04, GLIBC 2.39) is the one llama-server was compiled
# against -- mixing in a slimmer Ubuntu 22.04 nvidia/cuda image causes
# "GLIBC_2.38 not found" load-time failures (we paid for that lesson).
ARG LLAMACPP_VERSION=server-cuda
FROM ghcr.io/ggml-org/llama.cpp:${LLAMACPP_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# vast.ai's /.launch wrapper runs BEFORE our onstart_cmd and apt-installs
# openssh-server / tmux / rsync / software-properties-common at boot time.
# If we don't pre-install them and the rented host can't reach
# archive.ubuntu.com at boot (we observed this in the wild), /.launch loops
# forever on `ssh: command not found`. See LESSONS.md.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git gettext-base \
        netcat-openbsd jq sudo \
        openssh-server openssh-client openssh-sftp-server \
        rsync tmux software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Caddy with the duckdns DNS-01 plugin baked in, fetched as a static binary
# from the official Caddy build service (xcaddy under the hood).
RUN curl -fsSL "https://caddyserver.com/api/download?os=linux&arch=amd64&p=github.com%2Fcaddy-dns%2Fduckdns" \
        -o /usr/local/bin/caddy \
    && chmod +x /usr/local/bin/caddy \
    && /usr/local/bin/caddy version

# code-server: official install script picks the right .deb for the arch
# and installs it via apt. Pinned to a known-good version for reproducible
# builds; bump deliberately when needed.
ARG CODESERVER_VERSION=4.95.3
RUN curl -fsSL https://code-server.dev/install.sh \
        | sh -s -- --version "${CODESERVER_VERSION}" \
    && code-server --version

# opencode: the user types `opencode` in code-server's terminal and chats
# with the local llama-server. Installer drops the binary in
# /root/.opencode/bin -- we symlink to /usr/local/bin so it's on PATH for
# any shell (code-server's terminals open as root in the container).
RUN curl -fsSL https://opencode.ai/install | bash \
    && ln -sf /root/.opencode/bin/opencode /usr/local/bin/opencode \
    && opencode --version

# llama-server binary + libs are already at /app from the base image.
# Set the same env the existing self-contained backend assumes.
ENV LD_LIBRARY_PATH=/app
ENV HF_HOME=/workspace/hf_cache

# Bootstrap files. Templates live in /opt; onstart.sh renders them via
# envsubst with the env vars summon.py injects.
COPY onstart.sh           /opt/onstart.sh
COPY Caddyfile.template   /opt/Caddyfile.template
RUN chmod +x /opt/onstart.sh

EXPOSE 22 80 443 8080

CMD ["/opt/onstart.sh"]
