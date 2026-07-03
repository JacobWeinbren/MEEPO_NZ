#!/usr/bin/env bash
# =============================================================================
# setup.sh - environment setup for MEEPO (clean PyTorch) on a Blackwell server.
#
#   bash setup.sh
#
# Installs a Blackwell-capable PyTorch (CUDA 12.8 wheels — NVIDIA Blackwell,
# sm_100/sm_120, needs CUDA >= 12.8), the Python dependencies, and the package
# itself. Safe to re-run.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"     # cu128 (Blackwell) | cu130 | cu121 | cpu

echo "==> Python: $($PY --version)"
$PY -m pip install --upgrade pip wheel setuptools

# ---- PyTorch (GPU) ----------------------------------------------------------
# Blackwell GPUs require CUDA 12.8+; the cu128 index ships sm_100/sm_120 kernels.
if [ "$TORCH_CUDA" = "cpu" ]; then
  echo "==> Installing CPU-only PyTorch"
  $PY -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  echo "==> Installing PyTorch ($TORCH_CUDA)"
  $PY -m pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
      || { echo "!! ${TORCH_CUDA} wheels unavailable; falling back to default index"; \
           $PY -m pip install torch; }
fi

# ---- Python dependencies ----------------------------------------------------
echo "==> Installing Python requirements"
$PY -m pip install -r requirements.txt

# rasterio sometimes needs the binary wheel explicitly
$PY -c "import rasterio" 2>/dev/null || $PY -m pip install --only-binary=:all: rasterio || true

# ---- the package itself -----------------------------------------------------
# NOTE: MEEPO here is a CLEAN-PyTorch reimplementation. It deliberately does
# NOT depend on spconv / flash-attn / torch_scatter / timm (none build cleanly on
# Blackwell sm_120 and none can be CPU-smoke-tested). Do not install them.
echo "==> Installing meepo_nz (editable)"
$PY -m pip install -e .

# ---- sanity check -----------------------------------------------------------
echo "==> Verifying install"
$PY - <<'PYEOF'
import torch, numpy, scipy, matplotlib, laspy, yaml
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
import meepo_nz
from meepo_nz.models import build_meepo
from meepo_nz.utils.config import Config
m = build_meepo(Config())
print("MEEPO-L params:", f"{m.num_parameters():,}")
print("setup OK")
PYEOF

echo "==> Optional: run the CPU smoke test"
echo "    PYTHONPATH=. $PY scripts/smoke_test.py"
echo "==> Done. Next: bash run_all.sh   (or run scripts/01..05 individually)"
