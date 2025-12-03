from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from typing import Callable, List, Optional


def run_method_b(image_paths: List[str], out_dir: str, log: Callable[[str], None] = print) -> List[str]:
    """
    Nerfstudio pipeline:
      - copies inputs (keeping original filenames) into raw_images/
      - ns-process-data images -> processed/
      - ns-train nerfacto -> training/nerfacto/<run_id>/config.yml
      - ns-export (pointcloud, poisson, cameras) -> exports/<subdir>/

    Returns a list of generated artifact file paths.
    """
    outputs: List[str] = []
    image_paths = sorted(image_paths)
    def which_ok(name: str) -> bool:
        return shutil.which(name) is not None

    def stream(cmd: List[str]) -> int:
        log("[NeRF] $ " + " ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                if line := line.rstrip("\n"):
                    log(f"[NeRF] {line}")
            return proc.wait()
        except Exception as e:
            log(f"[NeRF] Failed to run: {e}")
            return 1

    def find_ns_config(train_root: str) -> Optional[str]:
        hits = []
        for root, _, files in os.walk(train_root):
            if "config.yml" in files:
                hits.append(os.path.join(root, "config.yml"))
        if not hits:
            return None
        hits.sort(key=lambda p: (("nerfacto" not in p), -p.count(os.sep)))
        return hits[0]

    # Sanity check
    if len(image_paths) < 3:
        log("[NeRF] Need at least 3 images.")
        return outputs
    for bin_name in ("ns-process-data", "ns-train", "ns-export"):
        if not which_ok(bin_name):
            log(f"[NeRF] '{bin_name}' not found in PATH.")
            return outputs

    # create workspace
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    ws = out_dir
    raw_dir = os.path.join(ws, "raw_images")
    proc_dir = os.path.join(ws, "processed")
    train_dir = os.path.join(ws, "training")
    export_dir = os.path.join(ws, "exports")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    # C) Copy images
    for src in image_paths:
        dst = os.path.join(raw_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            log(f"[NeRF] Failed to copy {src} -> {dst}: {e}")
            return outputs

    #process-data (NeRF-Studio’s own COLMAP workflow)
    if stream([
        "ns-process-data", "images",
        "--data", raw_dir,
        "--output-dir", proc_dir,
        "--sfm-tool", "colmap",
        "--matching-method", "sequential",
        "--refine-intrinsics",
        "--num-downscales", "2",
        "--gpu",
        "--verbose",
    ]) != 0:
        log("[NeRF] ns-process-data failed.")
        return outputs

    # 2) ns-train nerfacto
    exp_name = "nerfacto"
    if stream([
        "ns-train", "nerfacto",
        "--data", proc_dir,
        "--output-dir", train_dir,
        "--experiment-name", exp_name,
        "--timestamp", run_id,
        "--max-num-iterations", "20000",
        "--steps-per-eval-image", "3000",
        "--vis", "tensorboard",
        "--pipeline.model.predict-normals", "True",
    ]) != 0:
        log("[NeRF] ns-train failed.")
        return outputs

    #config.yml
    cfg = os.path.join(train_dir, exp_name, run_id, "config.yml")
    if not os.path.exists(cfg):
        cfg = find_ns_config(train_dir)
    if not cfg or not os.path.exists(cfg):
        log("[NeRF] Could not locate config.yml under training/.")
        return outputs
    log(f"[NeRF] Using config: {cfg}")

    #Export pointcloud
    pc_dir = os.path.join(export_dir, "pointcloud")
    os.makedirs(pc_dir, exist_ok=True)
    if stream([
        "ns-export", "pointcloud",
        "--load-config", cfg,
        "--output-dir", pc_dir,
        "--num-points", "8000000",
    ]) == 0:
        for cand in ("point_cloud.ply", "pointcloud.ply", "cloud.ply"):
            p = os.path.join(pc_dir, cand)
            if os.path.exists(p):
                outputs.append(p)
                log(f"[NeRF] Point cloud → {p}")
                break

    #Export Poisson mesh
    mesh_dir = os.path.join(export_dir, "poisson")
    os.makedirs(mesh_dir, exist_ok=True)
    if stream([
        "ns-export", "poisson",
        "--load-config", cfg,
        "--output-dir", mesh_dir,
        "--num-points", "8000000",
    ]) == 0:
        for cand in ("mesh.obj", "poisson_mesh.obj", "mesh_poisson.obj", "mesh.ply"):
            p = os.path.join(mesh_dir, cand)
            if os.path.exists(p):
                outputs.append(p)
                log(f"[NeRF] Poisson mesh → {p}")
                break

    #Export cameras
    cams_dir = os.path.join(export_dir, "cameras")
    os.makedirs(cams_dir, exist_ok=True)
    if stream([
        "ns-export", "cameras",
        "--load-config", cfg,
        "--output-dir", cams_dir,
    ]) == 0:
        for cand in ("cameras.json", "transforms.json"):
            p = os.path.join(cams_dir, cand)
            if os.path.exists(p):
                outputs.append(p)
                log(f"[NeRF] Cameras → {p}")
                break

    return outputs