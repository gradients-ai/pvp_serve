#!/usr/bin/env bash
# Serve the champion model with SGLang, matching G.O.D's PvP eval config
# (core/pvp/sglang_launch.build_base_command + qwen25 tool-call parser).
#
# The champion is a Qwen2.5-7B-Instruct lineage fine-tune, so the tool-call
# parser is qwen25. The model lives in a GATED HF repo — export HF_TOKEN first.
#
# Usage:
#   export HF_TOKEN=hf_...            # token with access to gradients-io-tournaments
#   ./scripts/serve_sglang.sh
set -euo pipefail

MODEL="${PVP_MODEL_REPO:-gradients-io-tournaments/tournament-tourn_358aca49563e214e_20260622-ac97eed9-69ff-4355-a012-2a9feaf3fd5f-5EEaxgnm}"
SERVED_NAME="${PVP_INFERENCE_MODEL:-champion}"
PORT="${SGLANG_PORT:-30000}"
SEED="${PVP_SEED:-0}"
TP="${SGLANG_TENSOR_PARALLEL_SIZE:-1}"
DTYPE="${SGLANG_DTYPE:-float16}"
PARSER="${SGLANG_TOOL_CALL_PARSER:-qwen25}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set — the champion repo is gated and will fail to download." >&2
fi
export HF_TOKEN="${HF_TOKEN:-}"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype "$DTYPE" \
  --enable-deterministic-inference --random-seed "$SEED" \
  --tool-call-parser "$PARSER"
