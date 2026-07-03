#!/usr/bin/env bash
# =============================================================================
# run_all.sh - end-to-end MEEPO pipeline on New Zealand LiDAR.
#
#   bash run_all.sh
#
# Stages: 01 download -> 02 build prev-year prior raster -> 04 preprocess ->
#         03 regional report -> 05 train. Override any setting via env vars:
#
#   DATA_ROOT=data/nz  BUDGET_GB=40  EPOCHS=500  BATCH_NUM=10  NUM_WORKERS=8  DEVICE=cuda \
#   CONFIG=configs/default.yaml  bash run_all.sh
#
# Set LIST_ONLY=1 to plan the download without fetching (dry run of stage 01).
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-data/nz}"
TILES_DIR="${TILES_DIR:-$DATA_ROOT/tiles}"
BUDGET_GB="${BUDGET_GB:-40}"
SOURCE="${SOURCE:-opentopography}"
EPOCHS="${EPOCHS:-500}"
NUM_WORKERS="${NUM_WORKERS:-8}"            # training DataLoader workers
CPU_WORKERS="${CPU_WORKERS:-$(nproc 2>/dev/null || echo 8)}"   # 01-04 parallelism (all cores)
DL_STREAMS="${DL_STREAMS:-16}"             # parallel download streams
BATCH_NUM="${BATCH_NUM:-6}"
DEVICE="${DEVICE:-cuda}"
NAME="${NAME:-meepo_nz_ground}"
OUT_DIR="${OUT_DIR:-runs}"
CONFIG="${CONFIG:-}"

CFG_ARG=""
[ -n "$CONFIG" ] && CFG_ARG="--config $CONFIG"

echo "############################################################"
echo "# MEEPO pipeline (clean PyTorch)"
echo "#   data_root = $DATA_ROOT"
echo "#   budget    = ${BUDGET_GB} GB   source = $SOURCE"
echo "#   epochs    = $EPOCHS  batch_num = $BATCH_NUM  train_workers = $NUM_WORKERS  cpu_workers = $CPU_WORKERS  dl_streams = $DL_STREAMS  device = $DEVICE"
echo "############################################################"

# ---- 01 download ------------------------------------------------------------
DL_ARGS=(--out "$DATA_ROOT" --budget-gb "$BUDGET_GB" --source "$SOURCE" --workers "$DL_STREAMS")
[ "${LIST_ONLY:-0}" = "1" ] && DL_ARGS+=(--list-only)
echo ">>> [01] download"
$PY scripts/01_download_data.py $CFG_ARG "${DL_ARGS[@]}"
[ "${LIST_ONLY:-0}" = "1" ] && { echo "LIST_ONLY set — stopping after planning."; exit 0; }

# ---- 02 previous-year DTM ---------------------------------------------------
echo ">>> [02] build previous-year CLASSIFICATION rasters (5-ch prior)"
$PY scripts/02_build_prior_raster.py $CFG_ARG --root "$DATA_ROOT" --workers "$CPU_WORKERS"

# ---- 04 preprocess ----------------------------------------------------------
# dl=0.10 (PTv3-native 10 cm) and in_radius=6 come from the config defaults; pass
# --dl / --in-radius to override. (--auto-dl would derive dl from data instead.)
echo ">>> [04] preprocess -> tiles"
$PY scripts/04_preprocess.py $CFG_ARG --root "$DATA_ROOT" --out "$TILES_DIR" --dl 0.1 --in-radius 6 --workers "$CPU_WORKERS"

# ---- 03 regional distribution report (on the preprocessed tiles) ------------
echo ">>> [03] regional sphere distribution"
$PY scripts/03_classify_and_sample.py $CFG_ARG --tile-dir "$TILES_DIR" || true

# ---- 05 train ---------------------------------------------------------------
echo ">>> [05] train"
$PY scripts/05_train.py $CFG_ARG --tiles "$TILES_DIR" --epochs "$EPOCHS" \
    --batch-num "$BATCH_NUM" --num-workers "$NUM_WORKERS" --device "$DEVICE" --out-dir "$OUT_DIR" --name "$NAME"

echo "############################################################"
echo "# Done. Per-epoch checkpoints, error images, classified LAZ"
echo "# and the training dashboard are under: $OUT_DIR/$NAME"
echo "############################################################"
