#!/bin/bash
# sbatch wrapper for GPU jobs. Excludes nodes whose NVIDIA driver is too old for our env
# (torch cu130 / vLLM 0.33 need driver >= 580, i.e. CUDA 13).
# Slurm doesn't expose the driver version, so this uses the kernel as a proxy: on 2026-11-03,
# nodes on kernel 6.8.0-147 had driver 580 and nodes on older kernels had CUDA 12.8 drivers.
# GPU jobs also call require_cuda13 (env.sh), so a wrong guess fails fast with a clear message.
# Usage: slurm/submit_gpu.sh [sbatch args] script.sub [script args]
set -euo pipefail
EXCL=$(scontrol show nodes | awk '/NodeName=/{n=$1; sub(/NodeName=/,"",n); g=0} /Gres=gpu/{g=1}
  /OS=Linux/{if (g && $2 !~ /^6\.8\.0-147/) print n}' | paste -sd,)
# Nodes that hung at NCCL/vLLM init or hit CUDA launch failures on 2026-11-05 (see analysis/teacher_tests_report.md);
# ma-compute-02: "CUDA unknown error" at torch init on 2026-11-06 (job 26943).
# badfellow: drained for GPU_Error_Detected on 2026-11-03; on 2026-11-06 job 28069 hung for 7h after vLLM
# warmup (GPUs at 100% util but low power, endless shm_broadcast warnings, LLM() never returned).
# nikola-compute-14: on 2026-11-06 higher-priority jobs preempted our gen jobs there 15+ times in 35 min,
# each within 2.5 min of start, so none got past model loading.
BAD=ellis-compute-02,nikola-compute-16,sablab-gpu-11,ma-compute-02,badfellow,nikola-compute-14
exec sbatch --exclude="$EXCL,$BAD" "$@"
