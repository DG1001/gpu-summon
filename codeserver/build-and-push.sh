#!/usr/bin/env bash
# Build and push the gpu-summon code-server image.
# Run this from the repo root after changes to codeserver/Dockerfile or onstart.sh.
#
# Requires: docker buildx, GHCR login (docker login ghcr.io).
# Or: skip this and let .github/workflows/build-codeserver-image.yml do it on push.

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/dg1001/gpu-summon-codeserver}"
TAG="${TAG:-latest}"
PLATFORM="${PLATFORM:-linux/amd64}"

cd "$(dirname "$0")"

echo "[build] $IMAGE:$TAG ($PLATFORM)"
docker buildx build \
    --platform "$PLATFORM" \
    --tag "$IMAGE:$TAG" \
    --push \
    .

echo "[done] pushed $IMAGE:$TAG"
echo "       use it via: gpu-summon --with-codeserver --code-image $IMAGE:$TAG ..."
