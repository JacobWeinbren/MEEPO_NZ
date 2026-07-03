# Running meepo_nz in ArcGIS Pro

## What the "deep learning extension" actually gives you here

Esri's **Deep Learning Libraries installer** (github.com/Esri/deep-learning-frameworks)
adds the *runtime* to Pro's `arcgispro-py3` env — and its manifest already contains
almost our whole stack: GPU **PyTorch**, **spconv 2.3.8** (prebuilt!), **laspy**,
**lazrs**, **rasterio**, scipy, boto3. Not included: **mamba-ssm** — and it is NOT needed: the SSM automatically runs the
built-in **chunked SSD scan** (Mamba-2 algorithm, pure torch, numerically identical to
the fused kernel's recurrence), so no Windows kernel build is required.

License reality check: the **Image Analyst / 3D Analyst extensions license Esri's own
geoprocessing tools** (Train Point Cloud Classification Model, etc.). Our MEEPO model is
NOT one of Esri's supported point-cloud architectures, so those tools / .dlpk packaging
are not used — and therefore **no extension license is required** to run this pipeline.
You need only: Pro + the (free) Deep Learning Libraries installer. Classified .laz output
loads into Pro as a normal LAS dataset (core functionality).

## One-time setup (Python Command Prompt, ships with Pro)

```bat
:: 0) Install ArcGIS Pro (3.6/3.7) + the matching Deep Learning Libraries MSI.
:: 1) Start menu > ArcGIS > Python Command Prompt, then:
conda create --clone arcgispro-py3 -n meepo
activate meepo
proswap meepo

:: 2) verify the Esri-provided runtime
python -c "import torch,spconv.pytorch,laspy,rasterio; print(torch.__version__, 'cuda', torch.cuda.is_available())"

:: 3) unzip meepo_nz.zip (e.g. to C:\meepo_nz) and register the package
cd /d C:\meepo_nz
python -m pip install --no-deps -e .
python -c "import meepo_nz, yaml, scipy, matplotlib; print('meepo_nz ok')"
```

If any import in step 3 fails, `python -m pip install <that package>` individually — do
NOT `pip install -r requirements.txt` wholesale into the Pro env (it may try to move
numpy/torch and break Esri's pinned set).

## Run (command line)

```bat
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /d C:\meepo_nz

:: official run: hand-crafted previous-year rasters (matched by file stem; partial coverage honest)
python scripts\02_build_prior_raster.py --input-dir D:\lidar\current --raster-dir D:\lidar\prior_rasters --root data\official --workers 8
python scripts\04_preprocess.py --root data\official --out data\official\tiles --dl 0.1 --in-radius 6 --workers 8

:: classify with a trained checkpoint (train elsewhere; copy runs\<name>\model_best.pt here)
python scripts\06_infer.py --checkpoint runs\meepo\model_best.pt --tiles data\official\tiles ^
  --input D:\lidar\current\CL2_XXXX.laz --out D:\out\CL2_XXXX_classified.laz --tta
```

## Run (inside Pro's UI)

Catalog pane > Folder Connections > add `C:\meepo_nz` > double-click **MeepoNZ.pyt**:

1. **Build Prior Raster** — current-year folder + hand-crafted raster folder (or prev-year LAZ folder).
2. **Preprocess Tiles** — the stage-02 workspace (or a raw LAS/LAZ folder) → tiles.
3. **Classify LAS/LAZ** — checkpoint + input .laz + tiles folder → classified .laz.

Add the output .laz to a LAS dataset and symbolize by Class Code (2 = ground).

## Training

Train on a Linux GPU box (`scripts/05_train.py`, see README) and copy
`runs/<name>/model_best.pt` + the tiles' `norm_stats.json` over. Training inside Pro is
possible from the Python Command Prompt but still not recommended: the chunked SSD scan
is much faster than the old fallback but remains slower than the fused CUDA kernel a
Linux box provides, and a multi-day train doesn't belong on a workstation running Pro.
