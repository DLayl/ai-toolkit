#!/usr/bin/env bash
# GH200 environment bootstrap for ai-toolkit.

set -euo pipefail

export AITK_GH200_ENABLE_UVM_HINTS="${AITK_GH200_ENABLE_UVM_HINTS:-1}"

: "${GH200_UVM_OVERSUBSCRIPTION_RATIO:=5.0}"
: "${GH200_UVM_ACCESS_PATTERN:=gpu_first}"

export PYTORCH_CUDA_ALLOC_CONF="use_uvm:True,uvm_oversubscription_ratio:${GH200_UVM_OVERSUBSCRIPTION_RATIO},uvm_access_pattern:${GH200_UVM_ACCESS_PATTERN}"

echo "[setup_gh200_env] ai-toolkit GH200 exports applied:"
echo "  AITK_GH200_ENABLE_UVM_HINTS=${AITK_GH200_ENABLE_UVM_HINTS}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
