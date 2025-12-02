from __future__ import annotations

import os
import sys
import subprocess
from typing import Optional

from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QFileDialog,
    QLabel, QCheckBox, QMessageBox, QTextEdit,
    QGroupBox, QFormLayout,
)

from .utils.paths import default_import_root
from .workers import ProcessorWorker

class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("2D to 3D Reconstruction Comparer")
        self.resize(900, 600)

        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(1)

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
        self.btn_open_out = QPushButton("Open Output Folder")

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
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_submit)
        btn_row.addWidget(self.btn_open_out)
        left.addLayout(btn_row)

        layout = QHBoxLayout()
        layout.addLayout(left, 2)
        layout.addWidget(self.log_view, 3)

        self.setLayout(layout)


        self.btn_add.clicked.connect(self.add_images)
        self.btn_clear.clicked.connect(self.clear_images)
        self.btn_submit.clicked.connect(self.on_submit)
        self.btn_open_out.clicked.connect(self.open_output_dir)

        self.output_dir = os.path.abspath("outputs")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def add_images(self) -> None:
        start_dir = default_import_root()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose images",
            start_dir,
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
        
    def _append_log_file(self, base_run_dir: str, text: str) -> None:
        try:
            with open(os.path.join(base_run_dir, "run.log"), "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass
        
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

        self._current_run_dir = None
        def tee_progress(msg: str):
            self.append_log(msg)
            if msg.startswith("[Run] Output root → "):
                self._current_run_dir = msg.split("→",1)[1].strip()
            if self._current_run_dir:
                self._append_log_file(self._current_run_dir, msg)
        worker.signals.progress.connect(tee_progress)
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
        
    def open_output_dir(self) -> None:
        path = self.output_dir
        try:
            if sys.platform.startswith("darwin"):
                subprocess.Popen(["open", path])
            elif os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            QMessageBox.information(self, "Open Folder", f"Folder: {path}\n\n({e})")
