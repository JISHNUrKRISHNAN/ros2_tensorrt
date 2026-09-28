#!/usr/bin/env bash
# Usage: build_engine.sh static.onnx model.engine [fp16|fp32]
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 STATIC_MODEL.onnx OUTPUT.engine [fp16|fp32]" >&2
  exit 2
fi
precision="${3:-fp16}"
case "$precision" in fp16|fp32) ;; *) echo 'Precision must be fp16 or fp32' >&2; exit 2;; esac
trtexec_bin="${TRTEXEC:-trtexec}"
if ! command -v "$trtexec_bin" >/dev/null 2>&1; then
  echo 'trtexec unavailable. Set TRTEXEC=/path/to/TensorRT/bin/trtexec.' >&2
  exit 1
fi
[[ -f "$1" ]] || { echo "Missing model: $1" >&2; exit 1; }
[[ ! -e "$2" ]] || { echo "Refusing to overwrite existing engine: $2" >&2; exit 1; }
mkdir -p "$(dirname "$2")"
args=("--onnx=$1" "--saveEngine=$2" --skipInference --memPoolSize=workspace:1024
      --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw)
if [[ "$precision" == fp16 ]]; then args+=(--fp16); else args+=(--noTF32); fi
# TensorRT 10 only. Static ONNX dimensions are required by this runtime.
exec "$trtexec_bin" "${args[@]}"
