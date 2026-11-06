"""Basic inference test for a chat model: answer a few prompts with thinking off, then one with thinking on.

Prints each answer, the finish reason, and throughput. Uses the sampling settings from the
Qwen3.8 model card. Text-only: the vision encoder is not loaded.

Usage: python scripts/infer_test.py <hf_id> [tensor_parallel_size]
"""
import sys
import time

from vllm import LLM, SamplingParams

PROMPTS = [
    "What is the capital of Australia? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number. Code only.",
    "Explain in two sentences why the sky is blue.",
]
THINK_PROMPT = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball?"

# Model-card recommendations (Qwen3.8): non-thinking vs thinking mode.
NO_THINK = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=512)
THINK = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=3072)


def chat(llm, prompts, params, **template_kwargs):
    t0 = time.time()
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params,
                    chat_template_kwargs=template_kwargs, use_tqdm=False)
    dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"\n--- {template_kwargs}: {len(prompts)} prompts, {ntok} tokens in {dt:.1f}s ({ntok / dt:.0f} tok/s)",
          flush=True)
    for p, o in zip(prompts, outs):
        print(f"\n> {p}\n{o.outputs[0].text.strip()}\n[finish={o.outputs[0].finish_reason}, "
              f"{len(o.outputs[0].token_ids)} tokens]", flush=True)
    return outs


def main():
    model = sys.argv[1]
    tp = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    t0 = time.time()
    llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=4096, gpu_memory_utilization=0.90,
              limit_mm_per_prompt={"image": 0, "video": 0})
    print(f"\n=== {model} loaded in {time.time() - t0:.0f}s (TP={tp})", flush=True)

    chat(llm, PROMPTS, NO_THINK, enable_thinking=False)
    out = chat(llm, [THINK_PROMPT], THINK, enable_thinking=True, reasoning_effort="low")[0].outputs[0]

    ok = out.finish_reason == "stop" and ("0.05" in out.text or "5 cents" in out.text)
    print("\nINFER_TEST_OK" if ok else "\nINFER_TEST_CHECK_OUTPUT (thinking answer missing $0.05 or hit max_tokens)",
          model)


if __name__ == "__main__":  # vLLM spawns workers that re-import this file
    main()
