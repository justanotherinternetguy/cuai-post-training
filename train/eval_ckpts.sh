#!/bin/bash
# Evaluate every checkpoint of one SFT run on the allocated GPU(s):
#   preference eval (eval/prefs_eval.py, chat format = the template saved with the checkpoint)
#     -> $OUT_DIR/eval/prefs/student_<student>__<run>__step<k>
#   next-token entropy on 500 held-out prompts (gen/generate.py)
#     -> $OUT_DIR/eval/gen/<student>__<run>__step<k>
# Usage: train/eval_ckpts.sh $OUT_DIR/ckpt/<student>/<run>
set -euo pipefail
RUN_DIR=${1%/}
RUN=$(basename "$RUN_DIR")
STUDENT=$(basename "$(dirname "$RUN_DIR")")
cd "$PROJECT_DIR"
for CK in $(ls -d "$RUN_DIR"/checkpoint-* | sort -t- -k2 -n); do
  STEP=${CK##*-}
  NAME="${STUDENT}__${RUN}__step${STEP}"
  echo "=== $NAME"
  python eval/prefs_eval.py --model "$CK" --name "student_$NAME" --format chat --tp 1
  python gen/generate.py --model "$CK" --name "$NAME" --splits heldout:500 --out-root "$OUT_DIR/eval/gen" --tp 1
done
echo "EVAL_CKPTS_OK"
