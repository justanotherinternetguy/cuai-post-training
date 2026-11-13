#!/bin/bash
# Evaluate every checkpoint of one SFT run on the allocated GPU(s):
#   preference eval (eval/prefs_eval.py, chat format = the template saved with the checkpoint)
#     -> $OUT_DIR/eval/prefs/student_<student>__<run>__step<k>
#   next-token entropy on 500 held-out prompts (gen/generate.py)
#     -> $OUT_DIR/eval/gen/<student>__<run>__step<k>
# With several GPUs, checkpoints run in parallel, one per GPU; each one's output goes to
# <log dir>/<job>_<name>.log. EVAL_MODES (default sample,logprob) is passed to prefs_eval.py --modes.
# Usage: train/eval_ckpts.sh $OUT_DIR/ckpt/<student>/<run>
set -euo pipefail
RUN_DIR=${1%/}
RUN=$(basename "$RUN_DIR")
STUDENT=$(basename "$(dirname "$RUN_DIR")")
MODES=${EVAL_MODES:-sample,logprob}
cd "$PROJECT_DIR"
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
LOGDIR="$PROJECT_DIR/logs"

eval_one() {  # checkpoint dir
  local CK=$1 STEP=${1##*-}
  local NAME="${STUDENT}__${RUN}__step${STEP}"
  echo "=== $NAME"
  python eval/prefs_eval.py --model "$CK" --name "student_$NAME" --format chat --tp 1 --modes "$MODES"
  python gen/generate.py --model "$CK" --name "$NAME" --splits heldout:500 --out-root "$OUT_DIR/eval/gen" --tp 1
}

CKS=($(ls -d "$RUN_DIR"/checkpoint-* | sort -t- -k2 -n))
if [ "${#GPUS[@]}" -le 1 ]; then
  for CK in "${CKS[@]}"; do eval_one "$CK"; done
else
  declare -A PIDS=()  # GPU index -> pid
  FAIL=0
  for CK in "${CKS[@]}"; do
    while :; do  # wait for a free GPU
      FREE=""
      for G in "${GPUS[@]}"; do
        P=${PIDS[$G]:-}
        if [ -z "$P" ]; then FREE=$G; break; fi
        if ! kill -0 "$P" 2>/dev/null; then wait "$P" || FAIL=1; unset "PIDS[$G]"; FREE=$G; break; fi
      done
      [ -n "$FREE" ] && break
      sleep 10
    done
    LOG="$LOGDIR/${SLURM_JOB_ID:-local}_${STUDENT}__${RUN}__step${CK##*-}.log"
    echo "=== ${CK##*/} on GPU $FREE, log $LOG"
    CUDA_VISIBLE_DEVICES=$FREE eval_one "$CK" > "$LOG" 2>&1 &
    PIDS[$FREE]=$!
  done
  for P in "${PIDS[@]}"; do wait "$P" || FAIL=1; done
  [ "$FAIL" -eq 0 ] || { echo "a checkpoint eval failed; see the logs above" >&2; exit 1; }
fi
echo "EVAL_CKPTS_OK"
