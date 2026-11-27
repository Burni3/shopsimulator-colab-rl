#!/usr/bin/env bash
# Zero-click fast restore of Slime + Megatron + SGLang + Pi + ShopSimulator from private Hugging Face Dataset repo.
set -euo pipefail

export WORK_ROOT="${WORK_ROOT:-/content/pi-slime-work}"
export HF_CACHE_REPO="${HF_CACHE_REPO:-your-hf-username/pi-slime-shopsimulator-cache}"

if [ -z "${HF_TOKEN:-}" ] && [ -f "/content/.hf_token" ]; then
  export HF_TOKEN="$(head -n 1 /content/.hf_token | tr -d '\r\n')"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set and /content/.hf_token not found" >&2
  exit 2
fi

if ! command -v zstd >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq zstd
fi

DL_DIR="/content/hf_cache_dl"
mkdir -p "${DL_DIR}" "${WORK_ROOT}"

echo "=== Downloading prebuilt environment bundles from private HF dataset (${HF_CACHE_REPO}) ==="
python3 - "${HF_CACHE_REPO}" "${DL_DIR}" <<'PY'
import os, sys, time
from huggingface_hub import hf_hub_download

repo_id, dl_dir = sys.argv[1:]
token = os.environ["HF_TOKEN"]
for fname in ("slime_env_bundle.tar.zst", "shopsimulator_ready.tar.zst"):
    t0 = time.time()
    p = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=fname,
        local_dir=dl_dir,
        token=token,
    )
    sz_mb = os.path.getsize(p) / (1024 * 1024)
    print(f"Downloaded {fname}: {sz_mb:.1f} MB in {time.time()-t0:.1f}s -> {p}", flush=True)
PY

echo "=== Unpacking Slime bundle to ${WORK_ROOT} ==="
tar -C "${WORK_ROOT}" -I "zstd -T0 -d" -xf "${DL_DIR}/slime_env_bundle.tar.zst"
rm -f "${DL_DIR}/slime_env_bundle.tar.zst"

echo "=== Unpacking ShopSimulator bundle to ${WORK_ROOT} ==="
tar -C "${WORK_ROOT}" -I "zstd -T0 -d" -xf "${DL_DIR}/shopsimulator_ready.tar.zst"
rm -f "${DL_DIR}/shopsimulator_ready.tar.zst"

cat > "${WORK_ROOT}/env.sh" <<'EOF'
export WORK_ROOT="/content/pi-slime-work"
export PROJECT_ROOT="${WORK_ROOT}/pi-slime-shopsimulator"
export SLIME_DIR="${PROJECT_ROOT}/slime"
export BASE_DIR="${WORK_ROOT}"
export MAMBA_ROOT_PREFIX="${WORK_ROOT}/micromamba"
export MAMBA_EXE="${WORK_ROOT}/bin/micromamba"
export MEGATRON_DIR="${WORK_ROOT}/Megatron-LM"
export SLIME_PYTHON="${MAMBA_ROOT_PREFIX}/envs/slime/bin/python"
export SHOP_ENV_PREFIX="${MAMBA_ROOT_PREFIX}/envs/shop_env"
export SHOP_PYTHON="${SHOP_ENV_PREFIX}/bin/python"
export SHOP_SIM_DIR="${WORK_ROOT}/ShopSimulator"
export PI_BIN="${WORK_ROOT}/pi-npm/node_modules/.bin/pi"
if [ -d "${MAMBA_ROOT_PREFIX}/envs/node22/bin" ]; then
  export PATH="${MAMBA_ROOT_PREFIX}/envs/node22/bin:${PATH}"
elif [ -d "/tools/node/bin" ]; then
  export PATH="/tools/node/bin:${PATH}"
fi
EOF

source "${WORK_ROOT}/env.sh"
echo "=== Applying verified runtime symlinks & pyarrow/numpy compatibility ==="
SLIME_ENV="${MAMBA_ROOT_PREFIX}/envs/slime"
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

echo "=== Verify restored environment ==="
"${PI_BIN}" --version
PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}" "${SLIME_PYTHON}" -c "import pyarrow, numpy, ml_dtypes, ray, sglang, torch, megatron.core, slime; print('HF_RESTORE_OK', torch.__version__, torch.version.cuda, 'numpy', numpy.__version__, 'pyarrow', pyarrow.__version__)"
