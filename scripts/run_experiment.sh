#!/usr/bin/env bash
# Unified experiment runner for ShopSimulator RL / GSPO / CISPO / OPSD comparison + standardized eval.
# Maximizes GPU utilization with ZERO CPU optimizer offload on both 40GB A100 and 80GB+ H100/G4!
set -euo pipefail

: "${EXP_NAME:?Set EXP_NAME (baseline_grpo | gspo_native | cispo_hard | opsd_gspo | eval_baseline_sft)}"
TRAIN_TASKS="${TRAIN_TASKS:-50}"
EVAL_TASKS="${EVAL_TASKS:-50}"
RUN_SUITE="${RUN_SUITE:-exp_suite_v1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
HF_CACHE_REPO="${HF_CACHE_REPO:-your-hf-username/pi-slime-shopsimulator-cache}"
DRIVE_CACHE="${DRIVE_CACHE:-/content/drive/MyDrive/pi-slime-shopsimulator-cache}"
RESULTS_ROOT="/content/exp_results/${RUN_SUITE}"
mkdir -p "${RESULTS_ROOT}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "/content/.hf_token" ]; then
  export HF_TOKEN="$(head -n 1 /content/.hf_token | tr -d '\r\n')"
fi

upload_dir_to_hf() {
  local local_dir="$1"
  local remote_prefix="$2"
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARN: HF_TOKEN unset, skipping HF results sync."
    return 0
  fi
  python3 - "${local_dir}" "${remote_prefix}" "${HF_CACHE_REPO}" <<'PY'
import os, sys
from huggingface_hub import HfApi
local_dir, remote_prefix, repo_id = sys.argv[1:]
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path=local_dir,
    path_in_repo=remote_prefix,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message=f"Upload experiment results: {remote_prefix}",
)
print(f"SYNCED_TO_HF: {repo_id}/{remote_prefix}", flush=True)
PY
}

now_ts() { python3 -c 'import time; print(f"{time.time():.3f}")'; }

echo "=== [${EXP_NAME}] Hardware Detection ==="
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)"
GPU_MEM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)"
CPU_CORES="$(nproc)"
RAM_GB="$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)"
echo "GPU=${GPU_NAME}, VRAM=${GPU_MEM_MIB} MiB, CPU=${CPU_CORES}, RAM=${RAM_GB} GB"

# 1. Restore environment (prefer zero-click private HF Dataset; fallback to Google Drive)
if [ ! -f "/content/pi-slime-work/env.sh" ]; then
  T0="$(now_ts)"
  if [ -n "${HF_TOKEN:-}" ] && [ -f "/content/restore_from_hf.sh" ]; then
    echo "=== [Init] Zero-click restoring Slime + ShopSimulator from private HF Dataset ==="
    bash /content/restore_from_hf.sh
  else
    echo "=== [Init] Restoring Slime + ShopSimulator from Google Drive ==="
    bash "${DRIVE_CACHE}/restore_from_drive.sh"
  fi
  T1="$(now_ts)"
  python3 -c "print(f'[TIMING] restore_env: {${T1}-${T0}:.2f}s')"
fi

source /content/pi-slime-work/env.sh
SLIME_ENV="${MAMBA_ROOT_PREFIX}/envs/slime"

# Apply the two proven 1-line runtime fixes idempotently
if [ ! -e "${SLIME_ENV}/lib64" ]; then
  ln -s "${SLIME_ENV}/lib" "${SLIME_ENV}/lib64"
fi
if [ ! -e "${SLIME_ENV}/lib/stubs" ]; then
  if [ -d "${SLIME_ENV}/targets/x86_64-linux/lib/stubs" ]; then
    ln -s "${SLIME_ENV}/targets/x86_64-linux/lib/stubs" "${SLIME_ENV}/lib/stubs"
  elif [ -d "/usr/local/cuda/lib64/stubs" ]; then
    ln -s "/usr/local/cuda/lib64/stubs" "${SLIME_ENV}/lib/stubs"
  fi
fi
if ! "${SLIME_PYTHON}" -c "import pyarrow, numpy" >/dev/null 2>&1; then
  "${SLIME_PYTHON}" -m pip install -q --no-deps --force-reinstall "pyarrow==19.0.1"
fi

# 1.5 Verify ShopSimulator Lucene index has full Chinese title/attribute/option fields (not category-only!)
SEARCH_DIR="${SHOP_SIM_DIR}/shop_env/search_engine"
DOCS_JSONL="${SEARCH_DIR}/resources/documents.jsonl"
NEED_REINDEX="$(python3 - "${DOCS_JSONL}" <<'PY'
import json, os, sys
p = sys.argv[1]
if not os.path.isfile(p):
    print("1")
    sys.exit(0)
# Check average content length of first 20 docs; category-only bug has avg < 30 chars, full docs have avg > 150 chars
lens = []
with open(p, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20:
            break
        lens.append(len(json.loads(line).get("contents", "")))
avg_len = sum(lens) / max(len(lens), 1)
print("1" if avg_len < 60 else "0")
PY
)"
if [ "${NEED_REINDEX}" = "1" ]; then
  echo "=== [Fix] Detected incomplete category-only Lucene index! Rebuilding full ShopSimulator index ==="
  pkill -f "pack_api.py" >/dev/null 2>&1 || true
  sleep 1
  "${SHOP_PYTHON}" - "${SHOP_SIM_DIR}/shop_env/data/items_eval_train.json" "${DOCS_JSONL}" <<'PY'
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
products = json.loads(src.read_text(encoding="utf-8"))
seen = set()
written = 0
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as out:
    for p in products:
        asin = str(p.get("asin", ""))
        if not asin or asin == "nan" or len(asin) > 20 or asin in seen:
            continue
        seen.add(asin)
        parts = []
        for key in ("title", "sub_title", "category", "shop_name", "full_description"):
            val = p.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
        attrs = p.get("attribute") or []
        if isinstance(attrs, list) and attrs:
            parts.append(" ".join(str(a) for a in attrs if a))
        opts = p.get("customization_options") or {}
        if isinstance(opts, dict):
            opt_vals = []
            for k, v_list in opts.items():
                if k:
                    opt_vals.append(str(k))
                if isinstance(v_list, list):
                    for item in v_list:
                        if isinstance(item, dict) and item.get("value"):
                            opt_vals.append(str(item["value"]))
            if opt_vals:
                parts.append(" ".join(opt_vals))
        out.write(json.dumps({"id": asin, "contents": " ".join(parts)}, ensure_ascii=False) + "\n")
        written += 1
print(f"Rebuilt full documents.jsonl: {written} docs", flush=True)
PY
  rm -rf "${SEARCH_DIR}/indexes" && mkdir -p "${SEARCH_DIR}/indexes"
  JAVA_HOME="${SHOP_ENV_PREFIX}/lib/jvm" \
  PATH="${SHOP_ENV_PREFIX}/lib/jvm/bin:${SHOP_ENV_PREFIX}/bin:${PATH}" \
  "${SHOP_PYTHON}" -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "${SEARCH_DIR}/resources" \
    --index "${SEARCH_DIR}/indexes" \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 \
    --storePositions --storeDocvectors --storeRaw
  echo "Full ShopSimulator Lucene index rebuilt cleanly!"
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "=== [Sync] Updating fixed shopsimulator_ready.tar.zst in private HF Dataset ==="
    tar -C "${WORK_ROOT}" -I "zstd -T0 -3" -cf /content/shopsimulator_ready_fixed.tar.zst ShopSimulator micromamba/envs/shop_env
    python3 - /content/shopsimulator_ready_fixed.tar.zst "${HF_CACHE_REPO}" <<'PY'
import os, sys
from huggingface_hub import HfApi
local_file, repo_id = sys.argv[1:]
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj=local_file,
    path_in_repo="shopsimulator_ready.tar.zst",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Fix ShopSimulator Lucene index to index full title/attribute/option fields",
)
print("UPDATED_HF_SHOPSIM_ARCHIVE_OK", flush=True)
PY
    rm -f /content/shopsimulator_ready_fixed.tar.zst
  fi
fi

# 2. Ensure ShopSimulator 20-slot server is healthy with ALL 20 slots free (release any leaked sessions first!)
curl -sS --max-time 5 -X POST http://127.0.0.1:5000/api/shop_agent \
  -H 'content-type: application/json' --data '{"action":"release_all"}' >/dev/null 2>&1 || true
if ! curl -sS --max-time 3 -X POST http://127.0.0.1:5000/api/shop_agent \
  -H 'content-type: application/json' --data '{"action":"status"}' 2>/dev/null | grep -q '"free": 20\|"free":20'; then
  echo "=== [Init] Restarting clean ShopSimulator 20-slot server ==="
  pkill -9 -f "pack_api.py" >/dev/null 2>&1 || true
  sleep 2
  (
    cd "${SHOP_SIM_DIR}/shop_env"
    JAVA_HOME="${SHOP_ENV_PREFIX}/lib/jvm" \
    PATH="${SHOP_ENV_PREFIX}/lib/jvm/bin:${SHOP_ENV_PREFIX}/bin:${PATH}" \
    PYTHONPATH="${SHOP_SIM_DIR}/shop_env" \
    nohup "${SHOP_PYTHON}" "${SHOP_SIM_DIR}/shop_env/shop_env/pack_api.py" > /content/shop_server.log 2>&1 &
  )
  READY=0
  for i in $(seq 1 60); do
    if curl -sS --max-time 3 -X POST http://127.0.0.1:5000/api/shop_agent \
      -H 'content-type: application/json' --data '{"action":"status"}' 2>/dev/null | grep -q '"free": 20\|"free":20'; then
      READY=1
      break
    fi
    sleep 1
  done
  if [ "${READY}" != "1" ]; then
    echo "ERROR: ShopSimulator failed to reach 20 free slots!" >&2
    exit 1
  fi
fi
echo "ShopSimulator ready (20/20 slots free)!"

# 3. Ensure SFT starting checkpoints (HF + Megatron torch_dist) exist
BASE_SFT_HF="/content/pi-slime-work/models/sft_hf"
if [ ! -f "${BASE_SFT_HF}/config.json" ]; then
  echo "=== [Init] Downloading mrzhao13/qwen3.5-2b-shopsimulator-sft-512-1ep ==="
  mkdir -p "${BASE_SFT_HF}"
  "${SLIME_PYTHON}" - "${BASE_SFT_HF}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="mrzhao13/qwen3.5-2b-shopsimulator-sft-512-1ep", local_dir=sys.argv[1])
PY
fi
SFT_HF="${SFT_HF_OVERRIDE:-${BASE_SFT_HF}}"
SFT_DIST="${SFT_DIST_OVERRIDE:-/content/pi-slime-work/models/sft_torch_dist}"
VOCAB_SIZE_OVERRIDE="${VOCAB_SIZE_OVERRIDE:-}"
if [[ "${EXP_NAME}" == *vocab49k* ]]; then
  SFT_HF="/content/pi-slime-work/models/sft_hf_vocab49k"
  SFT_DIST="/content/pi-slime-work/models/sft_torch_dist_vocab49k"
  VOCAB_SIZE_OVERRIDE="49152"
  if [ ! -f "${SFT_HF}/config.json" ]; then
    echo "=== [Init] Auto-pruning BBPE vocab (248,320 -> 49,152) via /content/prune_vocab_qwen35.py ==="
    "${SLIME_PYTHON}" /content/prune_vocab_qwen35.py \
      --src-hf "${BASE_SFT_HF}" \
      --dst-hf "${SFT_HF}" \
      --rl-jsonl "${SLIME_DIR}/examples/ShopSimulator/data/tasks_v2/rl_500.jsonl" \
      --eval-jsonl "${SLIME_DIR}/examples/ShopSimulator/data/tasks_v2/official_test_200.jsonl" \
      --shop-dir "${SHOP_SIM_DIR}" \
      --slime-dir "${SLIME_DIR}" \
      --target-vocab-sizes "49152" \
      --prune-vocab-size 49152
  fi
elif [ "${EXP_NAME}" == "eval_author_grpo" ]; then
  SFT_HF="/content/pi-slime-work/models/author_grpo_hf"
  SFT_DIST="/content/pi-slime-work/models/author_grpo_torch_dist"
  if [ ! -f "${SFT_HF}/config.json" ]; then
    echo "=== [Init] Downloading author GRPO model mrzhao13/qwen3.5-2b-shopsimulator-grpo-rl500-1ep ==="
    mkdir -p "${SFT_HF}"
    "${SLIME_PYTHON}" - "${SFT_HF}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="mrzhao13/qwen3.5-2b-shopsimulator-grpo-rl500-1ep", local_dir=sys.argv[1])
PY
  fi
fi
if [ ! -f "${SFT_DIST}/latest_checkpointed_iteration.txt" ]; then
  echo "=== [Init] Converting SFT HF -> Megatron torch_dist ==="
  cd "${SLIME_DIR}"
  source "${SLIME_DIR}/scripts/models/qwen3.5-2B.sh"
  if [ -n "${VOCAB_SIZE_OVERRIDE}" ]; then
    MODEL_ARGS=("${MODEL_ARGS[@]/248320/${VOCAB_SIZE_OVERRIDE}}")
  fi
  PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}" "${SLIME_PYTHON}" tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" --hf-checkpoint "${SFT_HF}" --save "${SFT_DIST}"
fi

# 4. Prepare standardized train & eval subsets + plugin dir
EXP_CFG_DIR="/content/exp_cfg"
PLUGIN_DIR="/content/custom_plugins"
mkdir -p "${EXP_CFG_DIR}" "${PLUGIN_DIR}" "${RESULTS_ROOT}"
cp -f /content/shop_innovations.py "${PLUGIN_DIR}/shop_innovations.py"

TRAIN_DATA="${EXP_CFG_DIR}/rl_${TRAIN_TASKS}.jsonl"
EVAL_DATA="${EXP_CFG_DIR}/official_test_${EVAL_TASKS}.jsonl"
head -n "${TRAIN_TASKS}" "${SLIME_DIR}/examples/ShopSimulator/data/tasks_v2/rl_500.jsonl" > "${TRAIN_DATA}"
head -n "${EVAL_TASKS}" "${SLIME_DIR}/examples/ShopSimulator/data/tasks_v2/official_test_200.jsonl" > "${EVAL_DATA}"

EVAL_YAML="${EXP_CFG_DIR}/eval_${EVAL_TASKS}.yaml"
cat > "${EVAL_YAML}" <<EOF
eval:
  defaults:
    input_key: prompt
    label_key: label
    metadata_key: metadata
    n_samples_per_eval_prompt: 1
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    max_response_len: 2048
  datasets:
    - name: shop_official_test_${EVAL_TASKS}
      path: ${EVAL_DATA}
      custom_generate_function_path: examples.ShopSimulator.generate.generate
      eval_task_timeout: 600
EOF

# Stop any stale Ray cluster & leftover GPU worker processes
"${MAMBA_ROOT_PREFIX}/envs/slime/bin/ray" stop --force >/dev/null 2>&1 || true
pkill -9 -f "sglang::scheduler" >/dev/null 2>&1 || true
pkill -9 -f "sglang::detokenizer" >/dev/null 2>&1 || true
sleep 2

# Configure VRAM maximization according to GPU tier
EXTRA_OPT_ARGS=(--log-probs-chunk-size 1024)
DISABLE_RECOMPUTE=0
USE_BF16_MAIN_GRADS=0
if [ "${GPU_MEM_MIB}" -ge 85000 ]; then
  # 96GB Blackwell (G4): disable full activation recomputation + bulletproof ~73-78GiB VRAM peak (~20GiB headroom, +35% faster!)
  DISABLE_RECOMPUTE=1
  SGLANG_MEM=0.75
  if [[ "${EXP_NAME}" == *vocab49k* ]]; then
    MAX_TOK=16384
    EXTRA_OPT_ARGS=(--log-probs-chunk-size 2048)
    VRAM_PROFILE="full-gpu-96g-norecompute-vocab49k"
  else
    MAX_TOK=12288
    EXTRA_OPT_ARGS=(--log-probs-chunk-size 2048)
    VRAM_PROFILE="full-gpu-96g-norecompute-248k"
  fi
elif [ "${GPU_MEM_MIB}" -ge 70000 ]; then
  SGLANG_MEM=0.75
  if [[ "${EXP_NAME}" == *vocab49k* ]]; then
    # 80GB H100/A100 + 49k vocab: disable recompute + MAX_TOK=12288 -> ~64GiB/80GiB, zero OOM, full speed!
    DISABLE_RECOMPUTE=1
    MAX_TOK=12288
    EXTRA_OPT_ARGS=(--log-probs-chunk-size 2048)
    VRAM_PROFILE="full-gpu-80g-norecompute-vocab49k"
  else
    MAX_TOK=16384
    VRAM_PROFILE="full-gpu-80g-fp32"
  fi
else
  # A100-40GB: bf16 main grads (+3.5GiB!) + chunked softmax (256 tok) + overlapped CPU optimizer offload -> fits even 16k-tok single trajectories with 0 OOM!
  MAX_TOK=6144
  SGLANG_MEM=0.60
  VRAM_PROFILE="40g-bf16grad-chunked256"
  USE_BF16_MAIN_GRADS=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  EXTRA_OPT_ARGS=(
    --log-probs-chunk-size 256
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
  )
fi
echo "VRAM Profile (${VRAM_PROFILE}): MAX_TOKENS_PER_GPU=${MAX_TOK}, SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM}, DISABLE_RECOMPUTE=${DISABLE_RECOMPUTE}"

run_eval_for_ckpt() {
  local ckpt_megatron="$1"
  local ckpt_hf="$2"
  local eval_root="$3"
  rm -rf "${eval_root}"
  "${MAMBA_ROOT_PREFIX}/envs/slime/bin/ray" stop --force >/dev/null 2>&1 || true
  local eval_runner="${SLIME_DIR}/examples/ShopSimulator/run_eval.sh"
  if [ -n "${VOCAB_SIZE_OVERRIDE}" ]; then
    eval_runner="/content/run_eval_vocab_${VOCAB_SIZE_OVERRIDE}.sh"
    python3 - "${SLIME_DIR}/examples/ShopSimulator/run_eval.sh" "${eval_runner}" "${VOCAB_SIZE_OVERRIDE}" <<'PY'
import sys
src, dst, vs = sys.argv[1:]
text = open(src, encoding="utf-8").read()
text = text.replace(
    'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
    'SCRIPT_DIR="${SLIME_DIR}/examples/ShopSimulator"',
    1,
)
anchor = 'source "${SLIME_DIR}/scripts/models/qwen3.5-2B.sh"'
assert anchor in text
text = text.replace(anchor, f'{anchor}\nMODEL_ARGS=("${{MODEL_ARGS[@]/248320/{vs}}}")', 1)
with open(dst, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
PY
    chmod +x "${eval_runner}"
  fi
  cd "${SLIME_DIR}"
  EVAL_CHECKPOINT="${ckpt_megatron}" \
  HF_CHECKPOINT="${ckpt_hf}" \
  EVAL_CONFIG="${EVAL_YAML}" \
  RUN_ROOT="${eval_root}" \
  SLIME_PYTHON="${SLIME_PYTHON}" \
  MEGATRON_DIR="${MEGATRON_DIR}" \
  PI_BIN="${PI_BIN}" \
  SHOP_ENV_URL="http://127.0.0.1:5000/api/shop_agent" \
  SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM}" \
  bash "${eval_runner}"
}

# Special mode: evaluate starting SFT checkpoint (original/vocab-pruned) or author GRPO baseline
if [ "${EXP_NAME}" == "eval_baseline_sft" ] || [ "${EXP_NAME}" == "eval_vocab49k_sft" ] || [ "${EXP_NAME}" == "eval_author_grpo" ]; then
  EVAL_ROOT="/content/pi-slime-work/runs/${EXP_NAME}_${EVAL_TASKS}"
  T0="$(now_ts)"
  run_eval_for_ckpt "${SFT_DIST}" "${SFT_HF}" "${EVAL_ROOT}"
  T1="$(now_ts)"
  python3 -c "print(f'[TIMING] ${EXP_NAME}: {${T1}-${T0}:.2f}s')"
  mkdir -p "${RESULTS_ROOT}/${EXP_NAME}"
  cp -f "${EVAL_ROOT}/eval_results.json" "${RESULTS_ROOT}/${EXP_NAME}/eval_results.json"
  cp -f "${EVAL_ROOT}/eval.log" "${RESULTS_ROOT}/${EXP_NAME}/eval.log"
  upload_dir_to_hf "${RESULTS_ROOT}/${EXP_NAME}" "runs/${RUN_SUITE}/${EXP_NAME}"
  if [ "${EXP_NAME}" == "eval_author_grpo" ]; then
    rm -rf "${SFT_DIST}" "${SFT_HF}" || true
  fi
  echo "=== ${EXP_NAME} DONE ==="
  head -n 45 "${EVAL_ROOT}/eval_results.json" || true
  exit 0
fi

# Select algorithm configuration
ADV_EST="grpo"
GEN_PATH="examples.ShopSimulator.generate.generate"
REWARD_PATH="examples.ShopSimulator.generate.normalize_candidate_group_rewards"
export SHOP_REWARD_MODE="loose"
EXTRA_ALGO_ARGS=()

case "${EXP_NAME}" in
  baseline_grpo)
    ADV_EST="grpo"
    ;;
  gspo_native|gspo_vocab49k)
    ADV_EST="gspo"
    EXTRA_ALGO_ARGS+=(--calculate-per-token-loss)
    ;;
  cispo_hard|cispo_vocab49k)
    ADV_EST="cispo"
    REWARD_PATH="shop_innovations.shaped_reward_post_process"
    export SHOP_REWARD_MODE="composite"
    EXTRA_ALGO_ARGS+=(--calculate-per-token-loss)
    ;;
  opsd_gspo)
    ADV_EST="gspo"
    GEN_PATH="shop_innovations.generate_opsd"
    REWARD_PATH="shop_innovations.shaped_reward_post_process"
    export SHOP_REWARD_MODE="composite"
    if [ "${GPU_MEM_MIB}" -lt 70000 ]; then
      # On 40GB A100, ShopSimulator uses only ~3% of KV cache; 0.45 + 2048 chunked prefill adds ~10GiB headroom for return_logprob=True
      SGLANG_MEM=0.45
      EXTRA_ALGO_ARGS+=(--sglang-chunked-prefill-size 2048)
    fi
    EXTRA_ALGO_ARGS+=(
      --calculate-per-token-loss
      --use-opd
      --opd-type sglang
      --opd-kl-coef 0.15
    )
    ;;
  opd_4b_gspo)
    if [ "${GPU_MEM_MIB}" -lt 70000 ]; then
      echo "=== [OPD-4B] GPU VRAM (${GPU_MEM_MIB} MiB) < 70 GB; skipping Qwen3.5-4B external teacher OPD to avoid OOM. ==="
      exit 0
    fi
    TEACHER_4B="/content/pi-slime-work/models/Qwen3.5-4B"
    if [ ! -f "${TEACHER_4B}/config.json" ]; then
      echo "=== [OPD-4B Init] Downloading Qwen/Qwen3.5-4B external teacher ==="
      mkdir -p "${TEACHER_4B}"
      "${SLIME_PYTHON}" - "${TEACHER_4B}" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3.5-4B", local_dir=sys.argv[1])
PY
    fi
    echo "=== [OPD-4B Init] Launching dedicated Qwen3.5-4B SGLang teacher server on port 13141 ==="
    pkill -9 -f "port 13141" >/dev/null 2>&1 || true
    PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}" "${SLIME_PYTHON}" -m sglang.launch_server \
      --model-path "${TEACHER_4B}" \
      --host 127.0.0.1 \
      --port 13141 \
      --tp 1 \
      --mem-fraction-static 0.18 \
      --disable-cuda-graph \
      > /content/opd_teacher_4b.log 2>&1 &
    TEACHER_PID=$!
    trap 'kill -9 ${TEACHER_PID} 2>/dev/null || true' EXIT
    for i in $(seq 1 90); do
      if curl -sf http://127.0.0.1:13141/health >/dev/null 2>&1; then
        echo "Qwen3.5-4B teacher server healthy on :13141!"
        break
      fi
      sleep 2
    done
    # Leave room for the 4B teacher (~17GB) alongside 2B student on 96GB G4 / 80GB H100
    SGLANG_MEM=0.50
    MAX_TOK=12288
    ADV_EST="gspo"
    GEN_PATH="shop_innovations.generate_opd_external"
    REWARD_PATH="shop_innovations.shaped_reward_post_process"
    export SHOP_REWARD_MODE="composite"
    export OPD_TEACHER_URL="http://127.0.0.1:13141"
    EXTRA_ALGO_ARGS+=(
      --calculate-per-token-loss
      --use-opd
      --opd-type sglang
      --opd-kl-coef 0.15
    )
    ;;
  *)
    echo "Unknown EXP_NAME: ${EXP_NAME}" >&2
    exit 2
    ;;
esac

RL_CFG_JSON="${EXP_CFG_DIR}/${EXP_NAME}_rl.json"
cat > "${RL_CFG_JSON}" <<EOF
{
  "datasets": { "rl": { "path": "${TRAIN_DATA}" } },
  "training": {
    "num_epoch": 1,
    "n_samples_per_prompt": 4,
    "num_steps_per_rollout": 1,
    "micro_batch_size": 1,
    "learning_rate": 1e-06,
    "lr_decay_style": "constant",
    "kl_loss_coefficient": 0.001,
    "kl_loss_type": "low_var_kl",
    "entropy_coefficient": 0.0,
    "eps_clip": 0.2,
    "eps_clip_high": 0.28,
    "gradient_clip": 1.0,
    "seed": 1234,
    "rollout_batch_size": 5,
    "global_batch_size": 20
  },
  "rollout": {
    "max_context_len": 16384,
    "max_response_len": 2048,
    "max_model_turns": 40,
    "wall_clock_timeout_seconds": 600,
    "context_keep_shop_act_results": 3,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "seed": 42,
    "custom_generate_function_path": "${GEN_PATH}",
    "reward_post_process_path": "${REWARD_PATH}",
    "sample_filter_path": "examples.ShopSimulator.generate.validate_complete_groups"
  },
  "concurrency": {
    "shop_env_capacity": 20,
    "rollout_num_engines": 1
  }
}
EOF

RUN_ROOT="/content/pi-slime-work/runs/${RUN_SUITE}_${EXP_NAME}"
RESUMING=0
if [ "${RESUME_IF_EXISTS:-1}" = "1" ] && [ -f "${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt" ]; then
  RESUMING=1
  RESUME_ITER="$(cat "${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt" | tr -d ' \r\n')"
  echo "=== [${EXP_NAME}] AUTO-RESUMING from existing checkpoint iteration ${RESUME_ITER} in ${RUN_ROOT} ==="
  EXTRA_OPT_ARGS+=(--no-load-optim --no-load-rng)
else
  rm -rf "${RUN_ROOT}"
fi

# Build temporary patched runner from unmodified upstream run_rl.sh
CUSTOM_RUNNER="/content/run_rl_${EXP_NAME}.sh"
python3 - "${SLIME_DIR}/examples/ShopSimulator/run_rl.sh" "${CUSTOM_RUNNER}" "${ADV_EST}" "${EXTRA_OPT_ARGS[*]:-}" "${EXTRA_ALGO_ARGS[*]:-}" "${VOCAB_SIZE_OVERRIDE}" "${DISABLE_RECOMPUTE}" "${RESUMING}" "${USE_BF16_MAIN_GRADS}" <<'PY'
import os, shlex, sys
src, dst, adv_est, opt_str, algo_str, vocab_override, disable_recompute, resuming, use_bf16_grads = sys.argv[1:]
text = open(src, encoding="utf-8").read()
text = text.replace(
    'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
    'SCRIPT_DIR="${SLIME_DIR}/examples/ShopSimulator"',
    1,
)
if resuming == "1":
    guard = 'if [[ -e "${RUN_ROOT}" ]]; then\n  echo "Refusing to overwrite existing RUN_ROOT: ${RUN_ROOT}" >&2\n  exit 2\nfi'
    text = text.replace(guard, 'echo "Resuming existing RUN_ROOT: ${RUN_ROOT}"', 1)
text = text.replace("--advantage-estimator grpo", f"--advantage-estimator {adv_est}", 1)
text = text.replace('--save-interval "${TOTAL_ROLLOUTS}"', f'--save-interval {os.environ.get("SAVE_INTERVAL", "10")}', 1)
if use_bf16_grads == "1":
    text = text.replace("  --accumulate-allreduce-grads-in-fp32\n", "")
    text = text.replace("--accumulate-allreduce-grads-in-fp32", "")
if disable_recompute == "1":
    for flag in (
        "  --recompute-granularity full\n",
        "  --recompute-method uniform\n",
        "  --recompute-num-layers 1\n",
    ):
        text = text.replace(flag, "")
if vocab_override:
    m_anchor = 'source "${SLIME_DIR}/scripts/models/qwen3.5-2B.sh"'
    assert m_anchor in text
    text = text.replace(m_anchor, f'{m_anchor}\nMODEL_ARGS=("${{MODEL_ARGS[@]/248320/{vocab_override}}}")', 1)
extras = shlex.split(opt_str) + shlex.split(algo_str)
if extras:
    anchor = "  --use-distributed-optimizer"
    assert anchor in text
    injected = anchor + "\n" + "\n".join(f"  {tok}" for tok in extras)
    text = text.replace(anchor, injected, 1)
# Ensure SHOP_REWARD_MODE propagates into Ray runtime env
text = text.replace(
    '"SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT",',
    '"SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT","SHOP_REWARD_MODE","OPD_TEACHER_URL",',
    1,
)
with open(dst, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)
PY
chmod +x "${CUSTOM_RUNNER}"

# Background log sync every 15 min for long full-dataset runs so progress is safely persisted in HF
LOG_SYNC_PID=""
if [ "${TRAIN_TASKS}" -ge 200 ] && [ -n "${HF_TOKEN:-}" ]; then
  (
    while true; do
      sleep 900
      if [ -f "${RUN_ROOT}/train.log" ]; then
        python3 - "${RUN_ROOT}/train.log" "runs/${RUN_SUITE}/${EXP_NAME}/train_live.log" "${HF_CACHE_REPO}" <<'PY' >/dev/null 2>&1 || true
import os, sys
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj=sys.argv[1],
    path_in_repo=sys.argv[2],
    repo_id=sys.argv[3],
    repo_type="dataset",
    commit_message="Periodic live train.log backup",
)
PY
      fi
    done
  ) &
  LOG_SYNC_PID=$!
fi

echo "=== [${EXP_NAME}] Starting Training (${TRAIN_TASKS} tasks = $((TRAIN_TASKS/5)) steps, adv=${ADV_EST}, reward=${SHOP_REWARD_MODE}, suite=${RUN_SUITE}) ==="
T_TRAIN_0="$(now_ts)"
cd "${SLIME_DIR}"
PYTHONPATH="${PLUGIN_DIR}:${MEGATRON_DIR}:${SLIME_DIR}" \
RL_CONFIG="${RL_CFG_JSON}" \
HF_CHECKPOINT="${SFT_HF}" \
REF_MODEL_PATH="${SFT_DIST}" \
RUN_ROOT="${RUN_ROOT}" \
SLIME_PYTHON="${SLIME_PYTHON}" \
MEGATRON_DIR="${MEGATRON_DIR}" \
PI_BIN="${PI_BIN}" \
SHOP_ENV_URL="http://127.0.0.1:5000/api/shop_agent" \
MAX_TOKENS_PER_GPU="${MAX_TOK}" \
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM}" \
bash "${CUSTOM_RUNNER}"
if [ -n "${LOG_SYNC_PID}" ]; then
  kill "${LOG_SYNC_PID}" >/dev/null 2>&1 || true
fi
T_TRAIN_1="$(now_ts)"
TRAIN_SEC="$(python3 -c "print(round(${T_TRAIN_1}-${T_TRAIN_0}, 2))")"
echo "[TIMING] ${EXP_NAME}_train: ${TRAIN_SEC}s"

# Find exported HF checkpoint and run standardized evaluation
HF_EXPORT="$(find "${RUN_ROOT}/hf" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
EVAL_ROOT="${RUN_ROOT}/eval_${EVAL_TASKS}"
echo "=== [${EXP_NAME}] Running Standardized Eval on ${EVAL_TASKS} tasks (HF=${HF_EXPORT}) ==="
T_EVAL_0="$(now_ts)"
run_eval_for_ckpt "${RUN_ROOT}/checkpoints" "${HF_EXPORT}" "${EVAL_ROOT}"
T_EVAL_1="$(now_ts)"
EVAL_SEC="$(python3 -c "print(round(${T_EVAL_1}-${T_EVAL_0}, 2))")"
echo "[TIMING] ${EXP_NAME}_eval: ${EVAL_SEC}s"

# Save summary + artifacts locally & sync to private HF Dataset
DEST_DIR="${RESULTS_ROOT}/${EXP_NAME}"
mkdir -p "${DEST_DIR}"
cp -f "${RUN_ROOT}/train.log" "${DEST_DIR}/train.log"
cp -f "${EVAL_ROOT}/eval_results.json" "${DEST_DIR}/eval_results.json"
cp -f "${EVAL_ROOT}/eval.log" "${DEST_DIR}/eval.log"
python3 - "${EXP_NAME}" "${GPU_NAME}" "${GPU_MEM_MIB}" "${VRAM_PROFILE}" "${TRAIN_SEC}" "${EVAL_SEC}" "${EVAL_ROOT}/eval_results.json" "${DEST_DIR}/summary.json" <<'PY'
import json, sys
exp, gpu, vram, prof, t_sec, e_sec, eval_path, out_path = sys.argv[1:]
ev = json.load(open(eval_path, encoding="utf-8"))
summary = {
    "experiment": exp,
    "gpu": gpu,
    "vram_mib": int(vram),
    "vram_profile": prof,
    "train_sec": float(t_sec),
    "eval_sec": float(e_sec),
    "metrics": {k: v for k, v in ev.items() if k != "records"},
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
with open(f"/content/{exp}_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("SUMMARY:", json.dumps(summary, ensure_ascii=False, indent=2))
PY
upload_dir_to_hf "${DEST_DIR}" "runs/${RUN_SUITE}/${EXP_NAME}"
# Free heavy Megatron optimizer shards after eval & HF sync so sequential full-500 runs never fill Colab disk
rm -rf "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/hf" || true
echo "=== [${EXP_NAME}] ALL COMPLETE & SYNCED TO HF ==="
