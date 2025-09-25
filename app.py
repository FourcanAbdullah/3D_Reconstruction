"""
PyQt6 app that lets a user:
  • Upload at least 3 images
  • Choose one or more processing methods via checkboxes
  • Click Submit to run the selected methods on the chosen images

This file includes stubbed method implementations you can replace with your own.
It also uses a background worker to keep the UI responsive while processing.

How to run:
  pip install PyQt6
  python app.py

(Optional) If you want the demo image outputs (contact sheet / edges), install:
  pip install pillow opencv-python
Both are optional; the app runs without them and will log a helpful note if missing.
"""
from __future__ import annotations

import shutil
import subprocess
import os
import sys
import time
from datetime import datetime
from typing import List, Callable, Tuple

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

# -----------------------------
# Optional demo dependencies
# -----------------------------
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


# -----------------------------
# Image processing stubs
# -----------------------------


def run_method_a(image_paths: List[str], output_dir: str, log: Callable[[str], None]) -> List[str]:
    """
    End-to-end COLMAP baseline:
      Sparse SfM  -> Dense MVS (fused.ply) -> Poisson mesh (poisson_mesh.ply)
    - Auto-detects GPU and enables CUDA if available
    - Uses CPU fallback otherwise
    - Returns key outputs: sparse model dir, TXT model, fused.ply, poisson mesh, report
    """
    outputs: List[str] = []
    log("[SfM] Starting COLMAP baseline (Sparse + Dense + Poisson)…")

    # --- sanity checks ---
    if len(image_paths) < 3:
        log("[SfM] Need at least 3 images.")
        return outputs
    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        log("[SfM] 'colmap' not found in PATH. Run inside the COLMAP dev-container.")
        return outputs

    # --- GPU auto-detect ---
    use_gpu = False
    try:
        p = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0:
            use_gpu = True
    except Exception:
        pass
    log(f"[SfM] {'GPU detected → enabling CUDA.' if use_gpu else 'No GPU detected → CPU mode.'}")

    # --- workspace layout ---
    ws = os.path.join(output_dir, "colmap_workspace")
    img_dir = os.path.join(ws, "images")
    sparse_dir = os.path.join(ws, "sparse")
    txt_dir = os.path.join(ws, "sparse_text")
    dense_dir = os.path.join(ws, "dense")
    mesh_dir = os.path.join(ws, "mesh")
    db_path = os.path.join(ws, "database.db")
    for d in (ws, img_dir, sparse_dir, txt_dir, dense_dir, mesh_dir):
        os.makedirs(d, exist_ok=True)

    # copy images into workspace (avoid cross-mount weirdness)
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
    max_img = os.getenv("COLMAP_MAX_IMAGE_SIZE", "2000")  # lower (e.g., 1600) for faster CPU runs

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

    # -------- Sparse pipeline --------
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

    # Export TXT (for inspection/versioning)
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

    # -------- Dense MVS --------
    # 1) Undistort
    if stream([
        "image_undistorter",
        "--image_path", img_dir,
        "--input_path", model0,
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
    ]) != 0:
        log("[Dense] image_undistorter failed.")
        return outputs

    # 2) PatchMatch stereo (geometry consistency helps)
    if stream([
        "patch_match_stereo",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ]) != 0:
        log("[Dense] patch_match_stereo failed.")
        return outputs

    # 3) Stereo fusion → fused point cloud
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

    # -------- Poisson meshing (via PyMeshLab) --------
    try:
        import pymeshlab as ml  # installed via requirements.txt
    except Exception:
        log("[Mesh] PyMeshLab not installed; skipping Poisson. (Add 'pymeshlab' to requirements.txt.)")
        log("[SfM] Done.")
        return outputs

    poisson_mesh = os.path.join(mesh_dir, "poisson_mesh.ply")
    report_path = os.path.join(mesh_dir, "poisson_report.txt")
    os.makedirs(mesh_dir, exist_ok=True)

    # Tunables via env (good for quick sweeps)
    depth = int(os.getenv("POISSON_DEPTH", "10"))                 # 8–12 typical
    samples_per_node = float(os.getenv("POISSON_SPN", "1.5"))
    point_weight = float(os.getenv("POISSON_POINT_WEIGHT", "4.0"))

    try:
        log("[Mesh] Running Screened Poisson meshing (PyMeshLab)…")
        ms = ml.MeshSet()
        ms.load_new_mesh(fused_ply)

        # Optional pre-cleaning (uncomment if you see artifacts)
        # ms.apply_filter('remove_duplicate_vertices')
        # ms.apply_filter('remove_unreferenced_vertices')

        ms.apply_filter(
            'surface_reconstruction_screened_poisson',
            depth=depth,
            samplespernode=samples_per_node,
            pointweight=point_weight
        )

        ms.save_current_mesh(poisson_mesh)

        # Basic mesh report (verts/faces + params)
        cur = ms.current_mesh()
        vnum = getattr(cur, "vertex_number", lambda: None)()
        fnum = getattr(cur, "face_number", lambda: None)()
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Poisson Mesh Report ===\n")
            f.write(f"Vertices: {vnum}\n")
            f.write(f"Faces:    {fnum}\n")
            f.write(f"Params: depth={depth}, samples_per_node={samples_per_node}, point_weight={point_weight}\n")
            f.write(f"Source:   {fused_ply}\n")

        outputs += [poisson_mesh, report_path]
        log(f"[Mesh] Poisson mesh → {poisson_mesh}")
        log(f"[Mesh] Report → {report_path}")
    except Exception as e:
        log(f"[Mesh] Poisson meshing failed: {e}")

    log("[SfM] Done.")
    return outputs




def run_method_b(image_paths: List[str], output_dir: str, log: Callable[[str], None]) -> List[str]:
    """
    Demo "Method B": runs Canny edge detection via OpenCV on each image (if available).
    If OpenCV is not installed, it logs actions only.

    Returns a list of generated output file paths (one per readable input).
    """
    outputs: List[str] = []
    log("[Method B] Starting...")
    if not image_paths:
        log("[Method B] No images provided.")
        return outputs

    os.makedirs(output_dir, exist_ok=True)

    for i, path in enumerate(image_paths, 1):
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f"{base}_edges.png")
        if not CV2_AVAILABLE:
            log("[Method B] OpenCV not installed. Skipping actual processing. Run 'pip install opencv-python' to enable.")
            log(f"[Method B] Would process: {os.path.basename(path)} → {out_path}")
            time.sleep(0.15)
            continue
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                log(f"[Method B] Could not open: {path}")
                continue
            edges = cv2.Canny(img, 100, 200)
            cv2.imwrite(out_path, edges)
            outputs.append(out_path)
            log(f"[Method B] Saved edges → {out_path} ({i}/{len(image_paths)})")
        except Exception as e:
            log(f"[Method B] Failed on {path}: {e}")
        time.sleep(0.05)

    log("[Method B] Done.")
    return outputs


# -----------------------------
# Worker infrastructure
# -----------------------------
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
            self.log("Processing started...")
            generated: List[str] = []

            if self.use_method_a:
                generated += run_method_a(self.image_paths, os.path.join(self.output_dir, "method_a"), self.log)

            if self.use_method_b:
                generated += run_method_b(self.image_paths, os.path.join(self.output_dir, "method_b"), self.log)

            if not (self.use_method_a or self.use_method_b):
                self.log("No methods selected. Nothing to do.")

            elapsed = (datetime.now() - started).total_seconds()
            summary = f"Processing complete in {elapsed:.2f}s. Generated {len(generated)} file(s)."
            self.signals.finished.emit(summary)
        except Exception as e:
            self.signals.error.emit(str(e))


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Multi-Image Processor (PyQt6)")
        self.resize(900, 600)

        self.thread_pool = QThreadPool.globalInstance()

        # Widgets
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.image_list.setMinimumHeight(220)

        self.btn_add = QPushButton("Add Images…")
        self.btn_clear = QPushButton("Clear List")

        self.chk_method_a = QCheckBox("Method A – 2D→3D reconstruction (COLMAP)")
        self.chk_method_b = QCheckBox("Method B – Canny edges (OpenCV)")
        self.chk_method_a.setChecked(True)

        self.lbl_count = QLabel("No images selected.")
        self.lbl_output = QLabel(f"Output folder: {os.path.abspath('outputs')}")
        self.lbl_output.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.btn_submit = QPushButton("Submit")
        self.btn_submit.setDefault(True)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Logs will appear here…")

        # Layouts
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

        # Connections
        self.btn_add.clicked.connect(self.add_images)
        self.btn_clear.clicked.connect(self.clear_images)
        self.btn_submit.clicked.connect(self.on_submit)

        # State
        self.output_dir = os.path.abspath("outputs")

    # -------------------------
    # UI helpers
    # -------------------------
    def add_images(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose images",
            "",
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
        # Autoscroll
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
