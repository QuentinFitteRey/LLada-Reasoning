#!/usr/bin/env bash
# Portable generation test launcher
#   GEN_MODEL      : model path/id (default: modelling_final)
#   GEN_PROMPT     : prompt text (default: "Hello")
#   GEN_MAX_TOKENS : max new tokens (default: 64)
#   GEN_ARGS       : additional generation args (comma key=value, e.g. temperature=0.7,top_p=0.9)
#   DRY_RUN=1      : print command only

set -euo pipefail

if command -v conda >/dev/null 2>&1 && [[ -d "$HOME/scratch/envs/llada" ]]; then
	conda activate "$HOME/scratch/envs/llada" || true
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

: "${GEN_MODEL:=modelling_final}"
: "${GEN_PROMPT:=Hello}"
: "${GEN_MAX_TOKENS:=64}"
: "${GEN_ARGS:=temperature=0.7}"

if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi || true
fi

read -r -d '' PYCODE <<'PYEOF'
import json, os
from init_model import load_model
from generation import generate_stream

model_path = os.environ.get('GEN_MODEL')
prompt = os.environ.get('GEN_PROMPT')
max_tokens = int(os.environ.get('GEN_MAX_TOKENS'))
extra_raw = os.environ.get('GEN_ARGS','')
extra = {}
for kv in filter(None, extra_raw.split(',')):
		if '=' in kv:
				k,v = kv.split('=',1)
				try:
						extra[k]=float(v)
				except ValueError:
						extra[k]=v

model, tokenizer = load_model(model_path, load_lora=False)
for out in generate_stream(model, tokenizer, prompt, max_new_tokens=max_tokens, extra_args_dict=extra):
		print(json.dumps(out, ensure_ascii=False))
PYEOF

export GEN_MODEL GEN_PROMPT GEN_MAX_TOKENS GEN_ARGS

CMD=(python - <<PYEOF
$PYCODE
PYEOF
)

echo "[GEN] MODEL=$GEN_MODEL PROMPT=$GEN_PROMPT MAX=$GEN_MAX_TOKENS ARGS=$GEN_ARGS" >&2
if [[ "${DRY_RUN:-0}" == "1" ]]; then
	printf '[DRY_RUN] '; printf '%q ' "${CMD[@]}"; echo
	exit 0
fi

"${CMD[@]}"

srun python fixed_generation_dual_cache.py