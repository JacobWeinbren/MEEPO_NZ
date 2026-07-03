# -*- coding: utf-8 -*-
"""MeepoNZ.pyt -- ArcGIS Pro Python toolbox for the meepo_nz ground-segmentation pipeline.

Runs the repo's own scripts inside the ArcGIS Pro python environment (arcgispro-py3 clone
with Esri's Deep Learning Libraries installed, which provides GPU PyTorch + spconv + laspy
+ rasterio). Add this repo folder as a Folder Connection in Pro's Catalog pane and
double-click MeepoNZ.pyt: three tools appear.

  1. Build Prior Raster   -> scripts/02_build_prior_raster.py  (hand-crafted raster / prev-year LAZ priors)
  2. Preprocess Tiles     -> scripts/04_preprocess.py          (LAS/LAZ folder -> model tiles)
  3. Classify LAS/LAZ     -> scripts/06_infer.py               (checkpoint -> classified .laz)

Training is deliberately NOT wrapped: run scripts/05_train.py from the Python Command
Prompt (a multi-day GPU job does not belong inside a geoprocessing tool), or train on a
Linux GPU box and copy runs/<name>/model_best.pt here.
"""
import os
import subprocess
import sys

import arcpy

_REPO = os.path.dirname(os.path.abspath(__file__))


def _run(messages, script, args):
    """Run a repo script with Pro's python, streaming its output into GP messages."""
    cmd = [sys.executable, os.path.join(_REPO, "scripts", script)] + [str(a) for a in args]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"
    messages.addMessage("> " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=_REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        messages.addMessage(line.rstrip())
    proc.wait()
    if proc.returncode != 0:
        raise arcpy.ExecuteError(f"{script} exited with code {proc.returncode} (see messages above)")


def _param(name, label, datatype, required=True, direction="Input", default=None):
    p = arcpy.Parameter(displayName=label, name=name, datatype=datatype,
                        parameterType=("Required" if required else "Optional"),
                        direction=direction)
    if default is not None:
        p.value = default
    return p


class Toolbox(object):
    def __init__(self):
        self.label = "MEEPO NZ ground segmentation"
        self.alias = "meeponz"
        self.tools = [BuildPriorRaster, PreprocessTiles, ClassifyLAZ]


class BuildPriorRaster(object):
    def __init__(self):
        self.label = "1. Build Prior Raster (stage 02)"
        self.description = ("Builds the 5-channel previous-year prior for each cloud and writes "
                            "<workspace>/manifest.json. Provide EITHER a folder of hand-crafted "
                            "previous-year rasters (GeoTIFF/ASC/.npz; matched to clouds by file stem; "
                            "partial coverage kept honest as coverage=0) OR a folder of previous-year "
                            "LAS/LAZ twins. With neither, clouds get no prior (prev-DTM zero-filled).")

    def getParameterInfo(self):
        return [
            _param("input_dir", "Current-year LAS/LAZ folder (OR use project tree below)", "DEFolder", required=False),
            _param("project_dir", "Project tree (subfolders each holding LAS + Previous DTM)", "DEFolder", required=False),
            _param("workspace", "Workspace folder (receives manifest.json + prior/)", "GPString"),
            _param("raster_dir", "Hand-crafted prior raster folder (optional)", "DEFolder", required=False),
            _param("prev_dir", "Previous-year LAS/LAZ folder (optional)", "DEFolder", required=False),
            _param("res", "Prior raster resolution (m, optional)", "GPDouble", required=False),
            _param("workers", "Worker processes", "GPLong", required=False, default=8),
        ]

    def execute(self, parameters, messages):
        p = {q.name: q.valueAsText for q in parameters}
        if not p.get("input_dir") and not p.get("project_dir"):
            raise arcpy.ExecuteError("Provide a LAS/LAZ folder or a project tree folder.")
        args = ["--root", p["workspace"]]
        if p.get("project_dir"):
            args += ["--project-dir", p["project_dir"]]
        else:
            args += ["--input-dir", p["input_dir"]]
        if p.get("raster_dir"):
            args += ["--raster-dir", p["raster_dir"]]
        elif p.get("prev_dir"):
            args += ["--prev-dir", p["prev_dir"]]
        if p.get("res"):
            args += ["--res", p["res"]]
        if p.get("workers"):
            args += ["--workers", p["workers"]]
        _run(messages, "02_build_prior_raster.py", args)


class PreprocessTiles(object):
    def __init__(self):
        self.label = "2. Preprocess Tiles (stage 04)"
        self.description = ("Voxel-subsamples LAS/LAZ into model tiles (+ norm_stats.json). Point it at "
                            "the stage-02 workspace to use the built priors, or directly at a LAS/LAZ "
                            "folder to tile without priors. dl and in-radius must match the values the "
                            "checkpoint was trained with (default dl=0.1, in-radius=6).")

    def getParameterInfo(self):
        return [
            _param("workspace", "Stage-02 workspace (has manifest.json) -- OR leave empty", "DEFolder", required=False),
            _param("input_dir", "LAS/LAZ folder (used only if no workspace manifest)", "DEFolder", required=False),
            _param("out_dir", "Output tiles folder", "GPString"),
            _param("dl", "Subsampling dl (m)", "GPDouble", default=0.1),
            _param("in_radius", "in-radius (m)", "GPDouble", default=6.0),
            _param("workers", "Worker processes", "GPLong", required=False, default=8),
        ]

    def execute(self, parameters, messages):
        p = {q.name: q.valueAsText for q in parameters}
        if not p.get("workspace") and not p.get("input_dir"):
            raise arcpy.ExecuteError("Provide the stage-02 workspace OR a LAS/LAZ input folder.")
        args = ["--out", p["out_dir"], "--dl", p["dl"], "--in-radius", p["in_radius"]]
        if p.get("workspace"):
            args = ["--root", p["workspace"]] + args
        if p.get("input_dir"):
            args = ["--input-dir", p["input_dir"]] + args
        if p.get("workers"):
            args += ["--workers", p["workers"]]
        _run(messages, "04_preprocess.py", args)


class ClassifyLAZ(object):
    def __init__(self):
        self.label = "3. Classify LAS/LAZ (stage 06)"
        self.description = ("Classifies one LAS/LAZ into ground / non-ground with a trained checkpoint "
                            "(runs/<name>/model_best.pt) and writes a classified .laz (ASPRS class 2 = "
                            "ground) that loads straight into a Pro LAS dataset. 'Tiles folder' is the "
                            "training tile dir -- it supplies norm_stats.json.")

    def getParameterInfo(self):
        return [
            _param("checkpoint", "Trained checkpoint (.pt)", "DEFile"),
            _param("input_laz", "Input LAS/LAZ", "DEFile"),
            _param("out_laz", "Output classified .laz", "GPString"),
            _param("tiles", "Tiles folder (norm_stats.json)", "DEFolder"),
            _param("prev_dtm", "Previous-year prior .npz (optional)", "DEFile", required=False),
            _param("tta", "Test-time augmentation (4x slower, slightly better)", "GPBoolean",
                   required=False, default=False),
            _param("device", "Device (cuda / cpu; optional)", "GPString", required=False),
            _param("epsg", "Output EPSG if source LAS lacks CRS (27700 = BNG)", "GPLong", required=False, default=27700),
        ]

    def execute(self, parameters, messages):
        p = {q.name: q.valueAsText for q in parameters}
        args = ["--checkpoint", p["checkpoint"], "--input", p["input_laz"],
                "--out", p["out_laz"], "--tiles", p["tiles"]]
        if p.get("prev_dtm"):
            args += ["--prev-dtm", p["prev_dtm"]]
        if (p.get("tta") or "").lower() == "true":
            args += ["--tta"]
        if p.get("device"):
            args += ["--device", p["device"]]
        if p.get("epsg"):
            args += ["--epsg", p["epsg"]]
        _run(messages, "06_infer.py", args)
        messages.addMessage("Done. Add the output .laz to a LAS dataset (or drag it into a scene) "
                            "and symbolize by Class Code: 2 = ground, 1 = non-ground.")
