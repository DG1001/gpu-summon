#!/usr/bin/env bash
# Build and push the gpu-summon full-stack image.
# Run this from the repo root after changes to xares/Dockerfile or onstart.sh.
#
# Requires: docker buildx, GHCR login (docker login ghcr.io).

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/dg1001/gpu-summon-fullstack}"
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
echo "       use it via: python summon.py --with-xares --xares-image $IMAGE:$TAG ..."
