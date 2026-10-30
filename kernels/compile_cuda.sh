#!/usr/bin/env bash
# Build MiniTorch's CUDA shared libraries into minitorch/cuda_kernels/.
# Run from the repository root on a machine with nvcc.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p minitorch/cuda_kernels
for name in combine softmax_kernel layernorm_kernel; do
  nvcc -o "minitorch/cuda_kernels/${name}.so" --shared "kernels/${name}.cu" -Xcompiler -fPIC
done
echo "built: $(ls minitorch/cuda_kernels)"
