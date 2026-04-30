#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="adversarialimage-ids:legacy"
CONTAINER_NAME="adversarialimage-ids-legacy"
PYTHONPATH_VALUE="/opt/caffe/python:/workspace/src/original/DeepDetector"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker exec -it "${CONTAINER_NAME}" bash
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker start "${CONTAINER_NAME}" >/dev/null
  docker exec -it "${CONTAINER_NAME}" bash
  exit 0
fi

docker run -dit \
  --name "${CONTAINER_NAME}" \
  -v "${PROJECT_ROOT}:/workspace" \
  -w /workspace \
  -e "PYTHONPATH=${PYTHONPATH_VALUE}" \
  "${IMAGE_NAME}" \
  bash >/dev/null

docker exec -it "${CONTAINER_NAME}" bash