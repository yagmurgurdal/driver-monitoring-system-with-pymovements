import argparse
import sys
import traceback
from pathlib import Path

import cv2

try:
    from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
    from PySide6.QtGui import QCloseEvent, QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QDoubleSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PySide6 is required for the desktop app. "
        "Install it with: .venv\\Scripts\\python.exe -m pip install PySide6"
    ) from exc

from realtime_driver_monitor import (
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_BUNDLE,
    MonitorFrameResult,
    RealtimeMonitorEngine,
    RuntimeConfig,
)

DEFAULT_DESKTOP_MODEL_BUNDLE = Path("results") / "random_forest_baseline" / "model_bundle.pkl"


def frame_to_pixmap(frame):
    if frame is None:
        return QPixmap()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class MetricCard(QFrame):
    def __init__(self, title: str, accent: str = "#5cc8ff"):
        super().__init__()
        self.setObjectName("MetricCard")
        self.setMinimumHeight(112)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("MetricTitle")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("MetricValue")
        self.value_label.setStyleSheet(f"color: {accent};")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        self.value_label.setText(value)


class MonitorWorker(QObject):
    frame_ready = Signal(object)
    session_ready = Signal(dict)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        model_bundle_path: Path,
        config: RuntimeConfig,
        camera_index: int,
        video_path: str,
        db_path: Path,
        disable_db: bool,
        enable_high_confidence_rule_gate: bool,
    ):
        super().__init__()
        self.model_bundle_path = Path(model_bundle_path)
        self.config = config
        self.camera_index = camera_index
        self.video_path = video_path
        self.db_path = Path(db_path)
        self.disable_db = disable_db
        self.enable_high_confidence_rule_gate = enable_high_confidence_rule_gate
        self._stop_requested = False
        self.engine = None

    @Slot()
    def run(self):
        try:
            self.engine = RealtimeMonitorEngine(
                model_bundle_path=self.model_bundle_path,
                config=self.config,
                camera_index=self.camera_index,
                video_path=self.video_path,
                db_path=self.db_path,
                disable_db=self.disable_db,
                enable_high_confidence_rule_gate=self.enable_high_confidence_rule_gate,
                render_overlay=False,
                resize_output=False,
            )
            self.session_ready.emit(
                {
                    "source": self.engine.source_description,
                    "fps": self.engine.fps,
                    "feature_set": self.engine.feature_set,
                    "session_id": self.engine.session_id,
                    "db_path": str(self.engine.db_path) if self.engine.db_path else "",
                }
            )
            while not self._stop_requested:
                result = self.engine.process_next_frame()
                if result is None:
                    break
                self.frame_ready.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())
        finally:
            if self.engine is not None:
                self.engine.close()
            self.finished.emit()

    def stop(self):
        self._stop_requested = True


class DriverMonitorWindow(QMainWindow):
    def __init__(self, auto_start_mode: str = "manual"):
        super().__init__()
        self.setWindowTitle("Driver Monitoring Desktop")
        self.resize(1480, 900)
        self.thread = None
        self.worker = None
        self.last_label = None
        self.auto_start_mode = auto_start_mode

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(22, 22, 22, 22)
        main_layout.setSpacing(18)

        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel("Driver Monitoring Desktop")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Live driver-state monitoring with baseline and PyMovements-supported inference."
        )
        subtitle.setObjectName("PageSubtitle")
        header.addWidget(title)
        header.addWidget(subtitle)
        main_layout.addLayout(header)

        controls = QFrame()
        controls.setObjectName("ControlPanel")
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setHorizontalSpacing(14)
        controls_layout.setVerticalSpacing(12)

        default_model_bundle = (
            DEFAULT_DESKTOP_MODEL_BUNDLE if DEFAULT_DESKTOP_MODEL_BUNDLE.exists() else DEFAULT_MODEL_BUNDLE
        )
        self.model_path_edit = QLineEdit(str(default_model_bundle))
        self.video_path_edit = QLineEdit("")
        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 10)
        self.camera_index_spin.setValue(0)
        self.window_sec_spin = QDoubleSpinBox()
        self.window_sec_spin.setRange(1.0, 10.0)
        self.window_sec_spin.setSingleStep(0.5)
        self.window_sec_spin.setValue(3.0)
        self.predict_every_spin = QSpinBox()
        self.predict_every_spin.setRange(1, 60)
        self.predict_every_spin.setValue(10)
        self.db_path_edit = QLineEdit(str(DEFAULT_DB_PATH))
        self.enable_db_check = QCheckBox("Enable SQLite logging")
        self.enable_db_check.setChecked(True)
        self.enable_alert_check = QCheckBox("Enable drowsiness alert")
        self.enable_alert_check.setChecked(True)
        self.enable_distraction_alert_check = QCheckBox("Enable distraction alert")
        self.enable_distraction_alert_check.setChecked(True)
        self.enable_gate_check = QCheckBox("Enable high-confidence rule gate")

        browse_model_btn = QPushButton("Browse Model")
        browse_model_btn.clicked.connect(self.browse_model)
        browse_video_btn = QPushButton("Browse Video")
        browse_video_btn.clicked.connect(self.browse_video)
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.start_video_btn = QPushButton("Start Video")
        self.start_video_btn.clicked.connect(self.start_video)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_monitor)
        self.stop_btn.setEnabled(False)

        controls_layout.addWidget(QLabel("Model Bundle"), 0, 0)
        controls_layout.addWidget(self.model_path_edit, 0, 1, 1, 3)
        controls_layout.addWidget(browse_model_btn, 0, 4)
        controls_layout.addWidget(QLabel("Video File"), 1, 0)
        controls_layout.addWidget(self.video_path_edit, 1, 1, 1, 3)
        controls_layout.addWidget(browse_video_btn, 1, 4)
        controls_layout.addWidget(QLabel("Camera Index"), 2, 0)
        controls_layout.addWidget(self.camera_index_spin, 2, 1)
        controls_layout.addWidget(QLabel("Window (sec)"), 2, 2)
        controls_layout.addWidget(self.window_sec_spin, 2, 3)
        controls_layout.addWidget(QLabel("Predict Every N Frames"), 2, 4)
        controls_layout.addWidget(self.predict_every_spin, 2, 5)
        controls_layout.addWidget(QLabel("Database Path"), 3, 0)
        controls_layout.addWidget(self.db_path_edit, 3, 1, 1, 3)
        controls_layout.addWidget(self.enable_db_check, 3, 4)
        controls_layout.addWidget(self.enable_alert_check, 3, 5)
        controls_layout.addWidget(self.enable_distraction_alert_check, 4, 4)
        controls_layout.addWidget(self.enable_gate_check, 4, 5)
        controls_layout.addWidget(self.start_camera_btn, 4, 0)
        controls_layout.addWidget(self.start_video_btn, 4, 1)
        controls_layout.addWidget(self.stop_btn, 4, 2)

        main_layout.addWidget(controls)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(18)
        main_layout.addLayout(content_layout, 1)

        self.video_label = QLabel("Click Start Camera or Start Video to begin a live session.")
        self.video_label.setObjectName("VideoPanel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(900, 540)
        content_layout.addWidget(self.video_label, 3)

        side_panel = QVBoxLayout()
        side_panel.setSpacing(14)
        content_layout.addLayout(side_panel, 2)

        self.state_card = MetricCard("Current State", accent="#ff9f43")
        self.conf_card = MetricCard("Confidence", accent="#5cc8ff")
        self.quality_card = MetricCard("Window Quality", accent="#7ef29a")
        self.perclos_card = MetricCard("PERCLOS", accent="#ffd166")
        self.ear_card = MetricCard("Mean EAR", accent="#c4a1ff")
        self.yaw_card = MetricCard("Yaw", accent="#ff7b89")

        cards_grid = QGridLayout()
        cards_grid.setHorizontalSpacing(12)
        cards_grid.setVerticalSpacing(12)
        cards = [
            self.state_card,
            self.conf_card,
            self.quality_card,
            self.perclos_card,
            self.ear_card,
            self.yaw_card,
        ]
        for idx, card in enumerate(cards):
            cards_grid.addWidget(card, idx // 2, idx % 2)
        side_panel.addLayout(cards_grid)

        info_frame = QFrame()
        info_frame.setObjectName("InfoPanel")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(16, 16, 16, 16)
        info_layout.setSpacing(8)

        self.source_value = QLabel("Awaiting live source")
        self.feature_set_value = QLabel("Not started")
        self.session_value = QLabel("Not started")
        self.db_value = QLabel("Not started")
        for title_text, value_label in [
            ("Source", self.source_value),
            ("Feature Set", self.feature_set_value),
            ("Session", self.session_value),
            ("DB Path", self.db_value),
        ]:
            title_label = QLabel(title_text)
            title_label.setObjectName("InfoTitle")
            value_label.setObjectName("InfoValue")
            info_layout.addWidget(title_label)
            info_layout.addWidget(value_label)
        side_panel.addWidget(info_frame)

        events_frame = QFrame()
        events_frame.setObjectName("InfoPanel")
        events_layout = QVBoxLayout(events_frame)
        events_layout.setContentsMargins(16, 16, 16, 16)
        events_layout.setSpacing(10)
        events_title = QLabel("Recent Events")
        events_title.setObjectName("SectionTitle")
        self.events_list = QListWidget()
        events_layout.addWidget(events_title)
        events_layout.addWidget(self.events_list)
        side_panel.addWidget(events_frame, 1)

        self.state_card.set_value("IDLE")
        self.conf_card.set_value("READY")
        self.quality_card.set_value("READY")
        self.perclos_card.set_value("READY")
        self.ear_card.set_value("READY")
        self.yaw_card.set_value("READY")

        self.statusBar().showMessage("Ready to start live monitor")
        self.apply_styles()
        if self.auto_start_mode != "manual":
            QTimer.singleShot(250, self.auto_start_monitor)

    def apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0c111b;
                color: #e7edf7;
                font-family: "Segoe UI";
                font-size: 14px;
            }
            #PageTitle {
                font-size: 34px;
                font-weight: 700;
                color: #f7fbff;
            }
            #PageSubtitle {
                font-size: 15px;
                color: #9fb2c8;
            }
            #ControlPanel, #InfoPanel, #MetricCard {
                background: #111a29;
                border: 1px solid #22324c;
                border-radius: 16px;
            }
            #VideoPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #121f33, stop:1 #0d1625);
                border: 1px solid #233554;
                border-radius: 20px;
                color: #93a4bb;
                font-size: 18px;
                padding: 8px;
            }
            #MetricTitle, #InfoTitle {
                color: #8ea2bc;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
            }
            #MetricValue {
                font-size: 24px;
                font-weight: 700;
                color: #f7fbff;
            }
            #InfoValue {
                color: #f7fbff;
                font-size: 13px;
            }
            #SectionTitle {
                color: #f7fbff;
                font-size: 18px;
                font-weight: 700;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QListWidget {
                background: #0b1422;
                border: 1px solid #24344f;
                border-radius: 10px;
                padding: 8px 10px;
                color: #f0f4fa;
            }
            QPushButton {
                background: #193354;
                border: 1px solid #2d4d78;
                border-radius: 12px;
                padding: 10px 14px;
                color: #f6fbff;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #22436d;
            }
            QPushButton:disabled {
                background: #162131;
                color: #72849b;
                border-color: #223042;
            }
            QCheckBox {
                color: #dce5f1;
            }
            QListWidget {
                min-height: 180px;
            }
            """
        )

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model bundle",
            str(Path(self.model_path_edit.text()).parent),
            "Pickle files (*.pkl);;All files (*)",
        )
        if path:
            self.model_path_edit.setText(path)

    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self.video_path_edit.setText(path)

    def build_config(self) -> RuntimeConfig:
        return RuntimeConfig(
            window_sec=float(self.window_sec_spin.value()),
            predict_every_frames=int(self.predict_every_spin.value()),
            min_confidence=0.45,
            min_confidence_margin=0.03,
            switch_confirmations=1,
            unknown_confirmations=3,
            drowsiness_alert_enabled=bool(self.enable_alert_check.isChecked()),
            distraction_alert_enabled=bool(self.enable_distraction_alert_check.isChecked()),
        )

    def auto_start_monitor(self):
        if self.thread is not None:
            return
        if self.auto_start_mode == "video" and self.video_path_edit.text().strip():
            self.start_video()
        else:
            self.start_camera()

    def start_camera(self):
        self.start_monitor(video_path="")

    def start_video(self):
        video_path = self.video_path_edit.text().strip()
        if not video_path:
            QMessageBox.warning(self, "Missing Video", "Select a video file first.")
            return
        self.start_monitor(video_path=video_path)

    def start_monitor(self, video_path: str):
        if self.thread is not None:
            QMessageBox.information(self, "Monitor Running", "Stop the current session first.")
            return

        model_path = Path(self.model_path_edit.text().strip())
        if not model_path.exists():
            QMessageBox.warning(self, "Missing Model", f"Model bundle not found:\n{model_path}")
            return

        self.events_list.clear()
        self.last_label = None
        self.state_card.set_value("STARTING")
        self.conf_card.set_value("SCANNING")
        self.quality_card.set_value("SCANNING")
        self.perclos_card.set_value("SCANNING")
        self.ear_card.set_value("SCANNING")
        self.yaw_card.set_value("SCANNING")
        self.video_label.setText("Initializing camera and model...")

        self.thread = QThread(self)
        self.worker = MonitorWorker(
            model_bundle_path=model_path,
            config=self.build_config(),
            camera_index=int(self.camera_index_spin.value()),
            video_path=video_path,
            db_path=Path(self.db_path_edit.text().strip()),
            disable_db=not bool(self.enable_db_check.isChecked()),
            enable_high_confidence_rule_gate=bool(self.enable_gate_check.isChecked()),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.session_ready.connect(self.on_session_ready)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

        self.start_camera_btn.setEnabled(False)
        self.start_video_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Monitoring started")

    @Slot(dict)
    def on_session_ready(self, payload: dict):
        self.source_value.setText(str(payload.get("source", "--")))
        self.feature_set_value.setText(str(payload.get("feature_set", "--")))
        self.session_value.setText(
            f"#{payload['session_id']}" if payload.get("session_id") is not None else "disabled"
        )
        self.db_value.setText(payload.get("db_path") or "disabled")
        self.events_list.addItem(
            f"Session started | source={payload.get('source')} | fps={payload.get('fps', 0):.1f}"
        )

    @Slot(object)
    def on_frame_ready(self, result: MonitorFrameResult):
        pixmap = frame_to_pixmap(result.display_frame)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

        self.state_card.set_value(result.label.upper())
        self.conf_card.set_value(f"{result.confidence:.0%}" if result.confidence >= 0 else "--")
        self.quality_card.set_value("USABLE" if result.usable else "LOW")

        perclos = None if not result.window_features else result.window_features.get("perclos")
        mean_ear = None if not result.window_features else result.window_features.get("mean_ear")
        yaw = result.live_row.get("yaw")

        self.perclos_card.set_value(f"{perclos:.3f}" if perclos is not None else "--")
        self.ear_card.set_value(f"{mean_ear:.3f}" if mean_ear is not None else "--")
        self.yaw_card.set_value(f"{yaw:.1f} deg" if yaw is not None else "--")

        if result.label != self.last_label:
            self.events_list.insertItem(
                0,
                f"{result.time_sec:7.2f}s | state changed to {result.label}",
            )
            self.last_label = result.label

        if result.alert_triggered:
            self.events_list.insertItem(
                0,
                f"{result.time_sec:7.2f}s | {result.alert_label or 'state'} alert triggered",
            )

        while self.events_list.count() > 40:
            self.events_list.takeItem(self.events_list.count() - 1)

        self.statusBar().showMessage(
            f"Frame {result.frame_index} | {result.label} | {result.time_sec:.2f}s"
        )

    @Slot(str)
    def on_worker_error(self, error_text: str):
        self.events_list.insertItem(0, "Runtime error occurred")
        self.statusBar().showMessage("Runtime error")
        self.state_card.set_value("ERROR")
        self.video_label.setText("Live session could not be started.")
        QMessageBox.critical(self, "Realtime Monitor Error", error_text)

    @Slot()
    def on_worker_finished(self):
        self.start_camera_btn.setEnabled(True)
        self.start_video_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Monitoring stopped")
        if self.video_label.pixmap() is None:
            self.video_label.setText("Live session stopped.")
        self.thread = None
        self.worker = None

    def stop_monitor(self):
        if self.worker is not None:
            self.worker.stop()
            self.statusBar().showMessage("Stopping monitor...")

    def closeEvent(self, event: QCloseEvent):
        self.stop_monitor()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(3000)
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Run the desktop driver monitoring app.")
    parser.add_argument(
        "--model-bundle",
        default=str(DEFAULT_MODEL_BUNDLE),
        help="Path to model_bundle.pkl produced by train_random_forest.py",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite database path used by the desktop monitor.",
    )
    parser.add_argument(
        "--startup-mode",
        choices=["camera", "video", "manual"],
        default="manual",
        help="Choose whether the app starts the camera immediately, starts the selected video, or waits manually.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = DriverMonitorWindow(auto_start_mode=args.startup_mode)
    window.model_path_edit.setText(str(args.model_bundle))
    window.db_path_edit.setText(str(args.db_path))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
