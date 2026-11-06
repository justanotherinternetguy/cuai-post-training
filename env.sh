# Shared settings. Every Slurm job and interactive shell should `source` this.
# Code lives in home; big files (weights, datasets, checkpoints) live on /share/cuvl.
export PROJECT_DIR=/home/qx235/unicorn
export ENV_DIR=/home/qx235/envs/unicorn
export DATA_ROOT=/share/cuvl/qx235
export HF_HOME=$DATA_ROOT/hf                         # model + dataset cache
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token  # keep the token private, not on the shared disk
export OUT_DIR=$DATA_ROOT/unicorn                    # generated data, checkpoints, eval results
export WANDB_DIR=$OUT_DIR/wandb
export VLLM_WORKER_MULTIPROC_METHOD=spawn            # vLLM fails if it forks after CUDA starts
export HF_XET_HIGH_PERFORMANCE=1
export PATH="$ENV_DIR/bin:$PATH"

# Call at the top of GPU jobs: stop early on nodes with a pre-CUDA-13 driver.
require_cuda13() {
  local drv
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  if [ "${drv%%.*}" -lt 580 ]; then
    echo "$(hostname -s) has NVIDIA driver $drv; this env needs >= 580. Submit with slurm/submit_gpu.sh." >&2
    exit 2
  fi
}
