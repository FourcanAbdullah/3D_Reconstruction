from __future__ import annotations

import shutil
import subprocess
import os
import sys
import time
from datetime import datetime
from typing import List, Callable, Tuple, Optional
import uuid
from PyQt6.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QLabel,
    QCheckBox,
    QMessageBox,
    QTextEdit,
    QGroupBox,
    QFormLayout,
)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import cv2  
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

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
    log(f"[SfM] {'GPU detected → enabling CUDA.' if use_gpu else 'No GPU detected → CPU mode.'}")

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
    max_img = os.getenv("COLMAP_MAX_IMAGE_SIZE", "2000")  # lower to lke 1600 for faster in CPU

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

    # sparse
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
        "--output_path", sparse_dir,
        f"--Mapper.num_threads={threads}",
    ]) != 0:
        log("[SfM] mapper failed.")
        return outputs

    model0 = os.path.join(sparse_dir, "0")
    if not os.path.isdir(model0):
        log("[SfM] No sparse model produced. Check overlap/EXIF.")
        return outputs

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

    #     delaunay_mesh = os.path.join(mesh_dir, "meshed-delaunay.ply")
    # log("[Mesh] (Optional) Running COLMAP Delaunay mesher…")
    # if stream([
    #     "delaunay_mesher",
    #     "--input_path", dense_dir,
    #     "--output_path", delaunay_mesh,
    # ]) == 0 and os.path.exists(delaunay_mesh):
    #     outputs.append(delaunay_mesh)
    #     log(f"[Mesh] Delaunay mesh → {delaunay_mesh}")

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

def run_method_b(image_paths: List[str], out_dir: str, log: Callable[[str], None] = print) -> List[str]:
    """
    Nerfstudio (NeRF) pipeline:
      - copies inputs (index-prefixed) into raw_images/
      - ns-process-data images -> processed/
      - ns-train nerfacto -> training/nerfacto/<run_id>/config.yml   (deterministic via --timestamp)
      - ns-export (pointcloud, poisson, cameras) -> exports/<subdir>/
    Returns a list of generated artifact file paths.
    """
    outputs: List[str] = []

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
        """Search for a config.yml under train_root; prefer deeper paths and ones containing 'nerfacto'."""
        hits = []
        for root, _, files in os.walk(train_root):
            if "config.yml" in files:
                hits.append(os.path.join(root, "config.yml"))
        if not hits:
            return None
        hits.sort(key=lambda p: (("nerfacto" not in p), -p.count(os.sep)))
        return hits[0]

    # sanity
    if len(image_paths) < 3:
        log("[NeRF] Need at least 3 images.")
        return outputs
    for bin_name in ("ns-process-data", "ns-train", "ns-export"):
        if not which_ok(bin_name):
            log(f"[NeRF] '{bin_name}' not found in PATH.")
            return outputs

    # workspace
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]  # used for --timestamp and pathing
    ws = out_dir  # caller creates a fresh run/method dir
    raw_dir = os.path.join(ws, "raw_images")
    proc_dir = os.path.join(ws, "processed")
    train_dir = os.path.join(ws, "training")
    export_dir = os.path.join(ws, "exports")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    # C) copy images with index prefix to avoid name collisions
    for i, src in enumerate(image_paths):
        base = os.path.basename(src)
        dst = os.path.join(raw_dir, f"{i:03d}__{base}")
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            log(f"[NeRF] Failed to copy {src} -> {dst}: {e}")
            return outputs

    # 1) ns-process-data (images → processed dataset)
    if stream([
        "ns-process-data", "images",
        "--data", raw_dir,
        "--output-dir", proc_dir,
        "--sfm-tool", "colmap",
        "--matching-method", "exhaustive",
        "--refine-intrinsics",
        "--gpu",
        "--verbose",
    ]) != 0:
        log("[NeRF] ns-process-data failed.")
        return outputs

    # 2) ns-train (nerfacto) with deterministic timestamp (B)
    exp_name = "nerfacto"
    if stream([
        "ns-train", "nerfacto",
        "--data", proc_dir,
        "--output-dir", train_dir,
        "--experiment-name", exp_name,
        "--timestamp", run_id,
        "--max-num-iterations", "8000",
        "--steps-per-eval-image", "2000",
        "--vis", "tensorboard",
        "--pipeline.model.predict-normals", "True",
        # deterministic run folder
        # Optional knobs (uncomment/tune as needed):
        # "--viewer.quit-on-train-completion", "True",
        # "--max-num-iterations", "20000",
    ]) != 0:
        log("[NeRF] ns-train failed.")
        return outputs

    # 3) Resolve config.yml (B)
    cfg = os.path.join(train_dir, exp_name, run_id, "config.yml")
    if not os.path.exists(cfg):
        # Fallback search in case NS version changes directory structure
        cfg = find_ns_config(train_dir)
    if not cfg or not os.path.exists(cfg):
        log("[NeRF] Could not locate config.yml under training/.")
        return outputs
    log(f"[NeRF] Using config: {cfg}")

    # 4) ns-export pointcloud (A: directory output, then pick produced file)
    pc_dir = os.path.join(export_dir, "pointcloud")
    os.makedirs(pc_dir, exist_ok=True)
    if stream([
        "ns-export", "pointcloud",
        "--load-config", cfg,
        "--output-dir", pc_dir,
        "--num-points", "2000000",
    ]) == 0:
        for cand in ("point_cloud.ply", "pointcloud.ply", "cloud.ply"):
            p = os.path.join(pc_dir, cand)
            if os.path.exists(p):
                outputs.append(p)
                log(f"[NeRF] Point cloud → {p}")
                break

    # 5) ns-export poisson mesh (A)
    mesh_dir = os.path.join(export_dir, "poisson")
    os.makedirs(mesh_dir, exist_ok=True)
    if stream([
        "ns-export", "poisson",
        "--load-config", cfg,
        "--output-dir", mesh_dir,
        "--num-points", "3000000",
    ]) == 0:
        for cand in ("mesh.obj", "poisson_mesh.obj", "mesh_poisson.obj", "mesh.ply"):
            p = os.path.join(mesh_dir, cand)
            if os.path.exists(p):
                outputs.append(p)
                log(f"[NeRF] Poisson mesh → {p}")
                break

    # 6) ns-export cameras (A)
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





class WorkerSignals(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)


class ProcessorWorker(QRunnable):
    def __init__(
        self,
        image_paths: List[str],
        use_method_a: bool,
        use_method_b: bool,
        output_dir: str,
    ) -> None:
        super().__init__()
        self.image_paths = image_paths
        self.use_method_a = use_method_a
        self.use_method_b = use_method_b
        self.output_dir = output_dir
        self.signals = WorkerSignals()

    def log(self, msg: str) -> None:
        self.signals.progress.emit(msg)

    def run(self) -> None:
        try:
            started = datetime.now()
            stamp = started.strftime("%Y%m%d-%H%M%S")        # ← run timestamp
            self.log("Processing started...")

            # Base run folder: outputs/run_YYYYmmdd-HHMMSS
            base_run_dir = os.path.join(self.output_dir, f"run_{stamp}")
            os.makedirs(base_run_dir, exist_ok=True)
            self.log(f"[Run] Output root → {base_run_dir}")

            generated: List[str] = []

            if self.use_method_a:
                out_a = os.path.join(base_run_dir, "method_a")
                generated += run_method_a(self.image_paths, out_a, self.log)

            if self.use_method_b:
                out_b = os.path.join(base_run_dir, "method_b")
                generated += run_method_b(self.image_paths, out_b, self.log)

            if not (self.use_method_a or self.use_method_b):
                self.log("No methods selected. Nothing to do.")

            elapsed = (datetime.now() - started).total_seconds()
            summary = f"Processing complete in {elapsed:.2f}s. Generated {len(generated)} file(s)."
            self.signals.finished.emit(summary)
        except Exception as e:
            self.signals.error.emit(str(e))



class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("2D to 3D Resonstruction Comparer")
        self.resize(900, 600)

        self.thread_pool = QThreadPool.globalInstance()


        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list.setMinimumHeight(220)

        self.btn_add = QPushButton("Add Images…")
        self.btn_clear = QPushButton("Clear List")

        self.chk_method_a = QCheckBox("Method A 2D→3D reconstruction (COLMAP)")
        self.chk_method_b = QCheckBox("Method B Nerf Studio Method")
        self.chk_method_a.setChecked(True)

        self.lbl_count = QLabel("No images selected.")
        self.lbl_output = QLabel(f"Output folder: {os.path.abspath('outputs')}")
        self.lbl_output.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.btn_submit = QPushButton("Submit")
        self.btn_submit.setDefault(True)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Logs will appear here…")

        top_btns = QHBoxLayout()
        top_btns.addWidget(self.btn_add)
        top_btns.addWidget(self.btn_clear)
        top_btns.addStretch(1)

        methods_group = QGroupBox("Select methods to run")
        fl = QFormLayout()
        fl.addRow(self.chk_method_a)
        fl.addRow(self.chk_method_b)
        methods_group.setLayout(fl)

        left = QVBoxLayout()
        left.addLayout(top_btns)
        left.addWidget(self.image_list)
        left.addWidget(self.lbl_count)
        left.addWidget(methods_group)
        left.addWidget(self.lbl_output)
        left.addWidget(self.btn_submit)

        layout = QHBoxLayout()
        layout.addLayout(left, 2)
        layout.addWidget(self.log_view, 3)

        self.setLayout(layout)


        self.btn_add.clicked.connect(self.add_images)
        self.btn_clear.clicked.connect(self.clear_images)
        self.btn_submit.clicked.connect(self.on_submit)


        self.output_dir = os.path.abspath("outputs")


    def add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose images",
            "/data/images",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        for f in files:
            if not f:
                continue
            item = QListWidgetItem(f)
            self.image_list.addItem(item)
        self.update_count()

    def clear_images(self) -> None:
        self.image_list.clear()
        self.update_count()

    def update_count(self) -> None:
        n = self.image_list.count()
        if n == 0:
            self.lbl_count.setText("No images selected.")
        else:
            self.lbl_count.setText(f"{n} image(s) selected.")

    def append_log(self, text: str) -> None:
        self.log_view.append(text)

        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def on_submit(self) -> None:
        n = self.image_list.count()
        if n < 3:
            QMessageBox.warning(self, "Need more images", "Please add at least 3 images before submitting.")
            return

        selected_methods = []
        if self.chk_method_a.isChecked():
            selected_methods.append("Method A")
        if self.chk_method_b.isChecked():
            selected_methods.append("Method B")

        if not selected_methods:
            reply = QMessageBox.question(
                self,
                "No methods selected",
                "No methods are checked. Submit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        image_paths = [self.image_list.item(i).text() for i in range(n)]
        self.append_log("\n===== New Run =====")
        self.append_log(f"Images: {n} | Methods: {', '.join(selected_methods) if selected_methods else 'None'}")

        worker = ProcessorWorker(
            image_paths=image_paths,
            use_method_a=self.chk_method_a.isChecked(),
            use_method_b=self.chk_method_b.isChecked(),
            output_dir=self.output_dir,
        )
        worker.signals.progress.connect(self.append_log)
        worker.signals.finished.connect(lambda msg: self.on_finished(msg))
        worker.signals.error.connect(lambda err: self.on_error(err))
        self.append_log(f"Base output folder: {self.output_dir}")
        self.btn_submit.setEnabled(False)
        self.append_log("Submitting job to background thread…")
        self.thread_pool.start(worker)

    def on_finished(self, msg: str) -> None:
        self.append_log(msg)
        self.btn_submit.setEnabled(True)

    def on_error(self, err: str) -> None:
        self.append_log(f"ERROR: {err}")
        self.btn_submit.setEnabled(True)


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
