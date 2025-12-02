from __future__ import annotations

import os
from datetime import datetime
from typing import List

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from .pipelines.colmap_pipeline import run_method_a
from .pipelines.nerfstudio_pipeline import run_method_b


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