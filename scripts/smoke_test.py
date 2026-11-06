"""End-to-end check of the toolchain on a tiny model (1 GPU, a few minutes).

Covers: vLLM generation, TRL SFT (a few steps), nnsight tracing, TransformerLens hooks.
"""
import os
import tempfile

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main():
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))

    # 1. vLLM generation
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, max_model_len=1024, gpu_memory_utilization=0.3)
    outs = llm.generate(["Question: What is your favorite color?\nAnswer:"], SamplingParams(max_tokens=10, n=4))
    print("[vllm]", [o.text for o in outs[0].outputs])
    del llm
    torch.cuda.empty_cache()

    # 2. TRL SFT for a few steps
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    ds = Dataset.from_list([{"messages": [{"role": "user", "content": "Pick a color."},
                                          {"role": "assistant", "content": "Blue."}]}] * 32)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = SFTTrainer(model=MODEL, train_dataset=ds,
                             args=SFTConfig(output_dir=tmp, max_steps=5, per_device_train_batch_size=4,
                                            logging_steps=1, report_to=[], bf16=True, save_strategy="no"))
        trainer.train()
    print("[trl] final loss", trainer.state.log_history[-1])

    # 3. nnsight: read a residual-stream activation
    from nnsight import LanguageModel
    lm = LanguageModel(MODEL, device_map="cuda")
    with lm.trace("The capital of France is"):
        h = lm.model.layers[5].output[0].save()
    print("[nnsight] layer 5 output shape", tuple(h.shape))

    # 4. TransformerLens 4.x: TransformerBridge replaces HookedTransformer.
    # enable_compatibility_mode() gives HookedTransformer-style weight processing and hook names.
    from transformer_lens import TransformerBridge
    tl = TransformerBridge.boot_transformers("gpt2", device="cuda")
    tl.enable_compatibility_mode()
    _, cache = tl.run_with_cache("Hello world")
    print("[transformer_lens] resid_post.0", tuple(cache["blocks.0.hook_resid_post"].shape))

    import wandb  # noqa: F401  (import check only; run `wandb login` once to log runs)
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
