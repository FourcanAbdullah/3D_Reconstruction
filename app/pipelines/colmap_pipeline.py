from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Callable, List


def run_method_a(image_paths: List[str], output_dir: str, log: Callable[[str], None]) -> List[str]:
    """
    COLMAP baseline:
      Sparse SfM  -> Dense MVS (fused.ply) -> Poisson mesh (poisson_mesh.ply)
    - Auto-detects GPU and enables CUDA if available
    - Uses CPU fallback otherwise
    - Returns key outputs: sparse model dir, TXT model, fused.ply, poisson mesh, report
    """
    outputs: List[str] = []
    log("[SfM] Starting COLMAP baseline (Sparse + Dense + Poisson)…")

    # edge cases
    if len(image_paths) < 3:
        log("[SfM] Need at least 3 images.")
        return outputs
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        log("[SfM] 'colmap' not found in PATH. Run inside the COLMAP dev-container.")
        return outputs

    #check gpu
    use_gpu = False
    try:
        p = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0:
            use_gpu = True
    except Exception:
        pass
    log(f"[SfM] {'GPU detected → enabling CUDA.' if use_gpu else 'No GPU detected → Will not work'}")

    # setup filesystem
    ws = os.path.join(output_dir, "colmap_workspace")
    img_dir = os.path.join(ws, "images")
    sparse_dir = os.path.join(ws, "sparse")
    txt_dir = os.path.join(ws, "sparse_text")
    dense_dir = os.path.join(ws, "dense")
    mesh_dir = os.path.join(ws, "mesh")
    db_path = os.path.join(ws, "database.db")
    for d in (ws, img_dir, sparse_dir, txt_dir, dense_dir, mesh_dir):
        os.makedirs(d, exist_ok=True)

    # copy images into file system
    log("[SfM] Copying input images…")
    copied = 0
    for pth in image_paths:
        try:
            shutil.copy2(pth, os.path.join(img_dir, os.path.basename(pth)))
            copied += 1
        except Exception as e:
            log(f"[SfM] Failed to copy {pth}: {e}")
    if copied < 3:
        log("[SfM] Fewer than 3 readable images after copy. Aborting.")
        return outputs

    threads = max(1, (os.cpu_count() or 2))
    max_img = os.getenv("COLMAP_MAX_IMAGE_SIZE", "2000")  

    def stream(args: List[str]) -> int:
        cmd = [colmap_bin] + args
        log("[SfM] $ " + " ".join(cmd))
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                if line := line.rstrip("\n"):
                    log(f"[SfM] {line}")
            return proc.wait()
        except Exception as e:
            log(f"[SfM] Failed to launch COLMAP: {e}")
            return 1

    # fresh DB
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except Exception:
        pass

    # sparse, feature extracter
    if stream([
        "feature_extractor",
        "--database_path", db_path,
        "--image_path", img_dir,
        f"--SiftExtraction.use_gpu={'true' if use_gpu else 'false'}",
        f"--SiftExtraction.num_threads={threads}",
        f"--SiftExtraction.max_image_size={max_img}",
    ]) != 0:
        log("[SfM] feature_extractor failed.")
        return outputs
    #sparse, feature matching
    if stream([
        "exhaustive_matcher",
        "--database_path", db_path,
        f"--SiftMatching.use_gpu={'true' if use_gpu else 'false'}",
        f"--SiftMatching.num_threads={threads}",
    ]) != 0:
        log("[SfM] exhaustive_matcher failed.")
        return outputs

    if stream([
        "mapper",
        "--database_path", db_path,
        "--image_path", img_dir,
        #--input_path, "path/to/your/model",
        "--output_path", sparse_dir,
        f"--Mapper.num_threads={threads}",
    ]) != 0:
        log("[SfM] mapper failed.")
        return outputs

    model0 = os.path.join(sparse_dir, "0")


    if not os.path.isdir(model0):
        log("[SfM] No sparse model produced. Check overlap/EXIF.")
        return outputs

    refined_model = os.path.join(sparse_dir, "0_ba")
    os.makedirs(refined_model, exist_ok=True)
    #bundle adjustment
    log("[SfM] Running bundle adjustment")
    if stream([
        "bundle_adjuster",
        "--input_path", model0,
        "--output_path", refined_model,
        "--BundleAdjustment.refine_focal_length", "1",    
        "--BundleAdjustment.refine_extra_params", "1",    
    ]) != 0:
        log("[SfM] bundle_adjuster failed; using original model")
        refined_model = model0   
    else:
        log(f"[SfM] Refined sparse model → {refined_model}")

    model0 = refined_model
    outputs.append(model0)
    log(f"[SfM] Sparse model → {model0}")

    # Export txt (for inspection/versioning)
    if stream([
        "model_converter",
        "--input_path", model0,
        "--output_path", txt_dir,
        "--output_type", "TXT",
    ]) == 0:
        for p in ("cameras.txt", "images.txt", "points3D.txt"):
            full = os.path.join(txt_dir, p)
            if os.path.exists(full):
                outputs.append(full)
        log(f"[SfM] TXT model saved → {txt_dir}")

    # dense mvs
    #Undistort
    if stream([
        "image_undistorter",
        "--image_path", img_dir,
        "--input_path", model0,
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
    ]) != 0:
        log("[Dense] image_undistorter failed.")
        return outputs

    # PatchMatch stereo 
    if stream([
        "patch_match_stereo",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ]) != 0:
        log("[Dense] patch_match_stereo failed.")
        return outputs

    #Stereo fusion 
    fused_ply = os.path.join(dense_dir, "fused.ply")
    if stream([
        "stereo_fusion",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--output_path", fused_ply,
    ]) != 0 or not os.path.exists(fused_ply):
        log("[Dense] stereo_fusion failed or fused.ply missing.")
        return outputs

    outputs.append(fused_ply)
    log(f"[Dense] Fused point cloud → {fused_ply}")
    
    # Poisson meshing
    os.makedirs(mesh_dir, exist_ok=True)
    poisson_mesh = os.path.join(mesh_dir, "meshed-poisson.ply")
    report_path = os.path.join(mesh_dir, "mesh_report.txt")

    log("[Mesh] Running COLMAP Poisson mesher…")
    if stream([
        "poisson_mesher",
        "--input_path", fused_ply,
        "--output_path", poisson_mesh,
    ]) != 0 or not os.path.exists(poisson_mesh):
        log("[Mesh] poisson_mesher failed or output missing.")
        log("[SfM] Done.")
        return outputs

    # Basic mesh report
    try:
        stats = os.stat(poisson_mesh)
        vnum = fnum = None
        try:
            # vertex/face count
            import trimesh
            m = trimesh.load_mesh(poisson_mesh)
            vnum, fnum = len(m.vertices), len(m.faces)
        except Exception:
            pass
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("----------COLMAP Poisson Mesh Report----------\n")
            f.write(f"Mesh path : {poisson_mesh}\n")
            f.write(f"Source PLY: {fused_ply}\n")
            f.write(f"Size (MB) : {stats.st_size / 1e6:.2f}\n")
            if vnum is not None and fnum is not None:
                f.write(f"Vertices  : {vnum}\n")
                f.write(f"Faces     : {fnum}\n")
            f.write(f"Modified  : {time.ctime(stats.st_mtime)}\n")
            f.write(f"Generated : COLMAP poisson_mesher\n")
        log(f"[Mesh] Poisson mesh → {poisson_mesh}")
        log(f"[Mesh] Report → {report_path}")
        outputs += [poisson_mesh, report_path]
    except Exception as e:
        log(f"[Mesh] Failed to create mesh report: {e}")

    log("[SfM] Done.")
    return outputs