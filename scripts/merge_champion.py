"""Merge a champion LoRA adapter into its base into full servable weights.

Mirrors the validator eval's merge (G.O.D validator/evaluation/eval_environment.py
::_merge_base_and_lora): load the base in fp16, attach the adapter, merge_and_unload,
save full weights + the adapter's tokenizer (it ships the chat template / added tokens).

The boss-round champion's eval base_chain is empty (PREVIOUS_WINNER tasks don't set a
starting_model_repo), so the base here is the foundation Qwen/Qwen2.5-7B-Instruct and we
apply the single champion adapter — exactly the model as it was crowned.

Usage:
  python scripts/merge_champion.py \
    --adapter gradients-io-tournaments/tournament-...-5EEaxgnm \
    --base Qwen/Qwen2.5-7B-Instruct \
    --out /opt/champion
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="HF repo / path of the champion LoRA adapter")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="foundation model repo/path")
    ap.add_argument("--out", default="/opt/champion", help="output dir for merged full weights")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[merge] loading base {args.base} on {args.device}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=args.device,
        trust_remote_code=True,
    )

    print(f"[merge] attaching adapter {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("[merge] merge_and_unload()")
    merged = model.merge_and_unload(safe_merge=False)

    print(f"[merge] saving merged weights -> {args.out}")
    merged.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")

    # The adapter repo carries the tokenizer + chat template the model was trained with.
    print("[merge] saving tokenizer from adapter repo")
    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tok.save_pretrained(args.out)

    print("[merge] done")


if __name__ == "__main__":
    main()
