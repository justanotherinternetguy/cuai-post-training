"""Load a model in vLLM and answer one prompt. Confirms the download is complete and loadable.

Usage: python scripts/check_model.py <hf_id> [tensor_parallel_size]
"""
import sys
import time

from vllm import LLM, SamplingParams


def main():
    model = sys.argv[1]
    tp = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    t0 = time.time()
    llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=4096,
              gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 0})
    print(f"\n=== {model} loaded in {time.time() - t0:.0f}s (TP={tp})", flush=True)
    params = SamplingParams(temperature=1.0, max_tokens=20, n=20)
    out = llm.chat([{"role": "user", "content": "Pick a random number between 1 and 10. Answer with just the number."}],
                   params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)[0]
    print("20 samples:", [o.text.strip() for o in out.outputs])
    print("CHECK_OK", model)


if __name__ == "__main__":  # vLLM spawns workers that re-import this file
    main()
