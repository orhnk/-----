from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .config import DetectionConfig
from .image_canvas import DrawingTool, ImageCanvas

_FACE_OVAL_LANDMARKS: Tuple[int, ...] = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)

_PERSISTED_FIELDS: Tuple[str, ...] = (
    "enable_face_mask",
    "enable_body_mask",
    "detection_confidence",
    "tracking_confidence",
    "segmentation_threshold",
    "segmentation_model_selection",
    "segmentation_smooth_factor",
    "segmentation_kernel_size",
    "mask_alpha",
    "mask_color",
    "border_thickness",
    "max_faces",
    "target_fps",
)

_MAX_KERNEL_SIZE = 31


class FaceDetectionWindow(QtWidgets.QMainWindow):
    def __init__(self, config: DetectionConfig) -> None:
        super().__init__()
        self.setWindowTitle("Live Face Detector")
        self.config = config

        self._segmentation_mask_buffer: Optional[np.ndarray] = None
        self._image_segmentation_mask_buffer: Optional[np.ndarray] = None
        self._face_mesh: mp.solutions.face_mesh.FaceMesh | None = None
        self._body_segmenter: mp.solutions.selfie_segmentation.SelfieSegmentation | None = None
        self._settings_file: Optional[Path] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._last_frame_time: Optional[float] = None
        self._dark_theme_enabled = False
        self._toggle_stream_action = None
        self._snapshot_action = None
        self._theme_action = None
        self._auto_paused_stream = False

        self._tab_widget: QtWidgets.QTabWidget | None = None
        self._camera_tab_index = -1
        self._image_tab_index = -1
        self._pdf_tab_index = -1

        self._image_canvas: ImageCanvas | None = None
        self._image_hint_label: QtWidgets.QLabel | None = None
        self._image_tool_combo: QtWidgets.QComboBox | None = None
        self._image_brush_spin: QtWidgets.QSpinBox | None = None
        self._image_color_button: QtWidgets.QPushButton | None = None
        self._image_save_button: QtWidgets.QPushButton | None = None
        self._image_clear_button: QtWidgets.QPushButton | None = None
        self._image_open_button: QtWidgets.QPushButton | None = None
        self._image_toolbar: QtWidgets.QWidget | None = None
        self._paste_shortcut: QtWidgets.QShortcut | None = None
        self._open_image_shortcut: QtWidgets.QShortcut | None = None
        self._save_image_shortcut: QtWidgets.QShortcut | None = None

        self._image_original_bgr: Optional[np.ndarray] = None
        self._image_last_directory: Optional[Path] = None

        self._load_persisted_settings()

        self._build_ui()
        self._create_toolbar()
        self._register_shortcuts()

        self._status_label = QtWidgets.QLabel("Initializing camera…")
        status_bar = QtWidgets.QStatusBar()
        status_bar.addWidget(self._status_label)
        self._fps_label = QtWidgets.QLabel("– fps")
        self._fps_label.setObjectName("fpsLabel")
        status_bar.addPermanentWidget(self._fps_label)
        self.setStatusBar(status_bar)

        self._sync_controls_from_config()

        self._camera = cv2.VideoCapture(self.config.camera_index, cv2.CAP_ANY)
        if not self._camera.isOpened():
            raise RuntimeError("Unable to access camera. Check that no other application is using it.")

        try:
            self._rebuild_face_mesh()
        except RuntimeError:
            if self._camera.isOpened():
                self._camera.release()
            raise

        try:
            self._rebuild_body_segmenter()
        except RuntimeError:
            if self._face_mesh is not None:
                self._face_mesh.close()
                self._face_mesh = None
            if self._camera.isOpened():
                self._camera.release()
            raise

        self._update_control_states()

        interval_ms = int(1000 / max(self.config.target_fps, 1))
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._timer.setInterval(interval_ms)
        self._start_streaming()

    def _build_ui(self) -> None:
        container = QtWidgets.QWidget(self)
        root_layout = QtWidgets.QVBoxLayout(container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._tab_widget = QtWidgets.QTabWidget(container)
        self._tab_widget.setDocumentMode(True)
        self._tab_widget.setTabsClosable(False)
        self._tab_widget.setMovable(False)
        self._tab_widget.currentChanged.connect(self._on_mode_changed)

        camera_tab = self._build_camera_tab(self._tab_widget)
        image_tab = self._build_image_tab(self._tab_widget)
        pdf_tab = self._build_pdf_tab(self._tab_widget)

        self._camera_tab_index = self._tab_widget.addTab(camera_tab, "Camera")
        self._image_tab_index = self._tab_widget.addTab(image_tab, "Image")
        self._pdf_tab_index = self._tab_widget.addTab(pdf_tab, "PDF")
        self._tab_widget.setCurrentIndex(self._camera_tab_index)

        root_layout.addWidget(self._tab_widget)
        self.setCentralWidget(container)

    def _build_camera_tab(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, tab)
        splitter.setChildrenCollapsible(False)

        video_container = QtWidgets.QFrame(splitter)
        video_container.setObjectName("videoContainer")
        video_layout = QtWidgets.QVBoxLayout(video_container)
        video_layout.setContentsMargins(16, 16, 16, 16)
        video_layout.setSpacing(12)

        self._video_label = QtWidgets.QLabel("Camera initializing…", video_container)
        self._video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setWordWrap(True)
        self._video_label.setStyleSheet(
            "QLabel { background-color: #10161c; color: #9fb2c9; border: 1px solid #1f2a33; border-radius: 12px; }"
        )
        video_layout.addWidget(self._video_label, stretch=1)

        hint_label = QtWidgets.QLabel(
            "Use the controls to toggle masks, adjust precision, and change the overlay appearance in real time.",
            video_container,
        )
        hint_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        hint_label.setWordWrap(True)
        hint_label.setObjectName("hintLabel")
        video_layout.addWidget(hint_label)

        splitter.addWidget(video_container)

        controls_scroll = QtWidgets.QScrollArea(splitter)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setObjectName("controlScroll")
        control_panel = self._create_control_panel(controls_scroll)
        controls_scroll.setWidget(control_panel)
        splitter.addWidget(controls_scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([880, 320])

        layout.addWidget(splitter)
        return tab

    def _build_image_tab(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)

        header = QtWidgets.QLabel("Image studio", tab)
        header.setStyleSheet("font-size: 20px; font-weight: 600;")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(header)

        self._image_hint_label = QtWidgets.QLabel(
            "Paste any image with Ctrl+V or drop files here to instantly anonymize faces and bodies.",
            tab,
        )
        self._image_hint_label.setWordWrap(True)
        self._image_hint_label.setStyleSheet("color: #5c6a79;")
        layout.addWidget(self._image_hint_label)

        toolbar_widget = QtWidgets.QWidget(tab)
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar_widget)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(10)

        self._image_open_button = QtWidgets.QPushButton("Open image…", toolbar_widget)
        self._image_open_button.clicked.connect(self._on_image_open_clicked)
        toolbar_layout.addWidget(self._image_open_button)

        paste_button = QtWidgets.QPushButton("Paste now", toolbar_widget)
        paste_button.clicked.connect(self._handle_paste_request)
        toolbar_layout.addWidget(paste_button)

        self._image_save_button = QtWidgets.QPushButton("Export…", toolbar_widget)
        self._image_save_button.clicked.connect(self._on_image_save_clicked)
        toolbar_layout.addWidget(self._image_save_button)

        self._image_clear_button = QtWidgets.QPushButton("Clear overlay", toolbar_widget)
        self._image_clear_button.clicked.connect(self._on_image_clear_clicked)
        toolbar_layout.addWidget(self._image_clear_button)

        undo_button = QtWidgets.QPushButton("Undo", toolbar_widget)
        undo_button.clicked.connect(self._on_image_undo_clicked)
        toolbar_layout.addWidget(undo_button)

        toolbar_layout.addSpacing(12)

        self._image_tool_combo = QtWidgets.QComboBox(toolbar_widget)
        self._image_tool_combo.addItem("Brush", DrawingTool.BRUSH)
        self._image_tool_combo.addItem("Rectangle", DrawingTool.RECTANGLE)
        self._image_tool_combo.currentIndexChanged.connect(self._on_image_tool_changed)
        toolbar_layout.addWidget(self._image_tool_combo)

        self._image_brush_spin = QtWidgets.QSpinBox(toolbar_widget)
        self._image_brush_spin.setRange(1, 200)
        self._image_brush_spin.setValue(48)
        self._image_brush_spin.setSuffix(" px")
        self._image_brush_spin.valueChanged.connect(self._on_image_brush_size_changed)
        toolbar_layout.addWidget(self._image_brush_spin)

        self._image_color_button = QtWidgets.QPushButton("Brush color…", toolbar_widget)
        self._image_color_button.clicked.connect(self._on_image_color_clicked)
        toolbar_layout.addWidget(self._image_color_button)

        toolbar_layout.addStretch(1)

        layout.addWidget(toolbar_widget)
        self._image_toolbar = toolbar_widget

        canvas_container = QtWidgets.QFrame(tab)
        canvas_container.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        canvas_container.setObjectName("imageCanvasFrame")
        canvas_layout = QtWidgets.QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)

        self._image_canvas = ImageCanvas(canvas_container)
        self._image_canvas.updated.connect(self._update_image_controls_state)
        self._image_canvas.dropped.connect(self._on_image_dropped)
        canvas_layout.addWidget(self._image_canvas, stretch=1)
        layout.addWidget(canvas_container, stretch=1)

        footer = QtWidgets.QLabel(
            "Tip: tune detection settings in the Camera tab—the same parameters apply to image processing.",
            tab,
        )
        footer.setWordWrap(True)
        footer.setStyleSheet("color: #5c6a79; font-size: 12px;")
        layout.addWidget(footer)

        # Initialize tool defaults
        self._image_canvas.set_pen_color(self._initial_brush_color())
        self._image_canvas.set_pen_width(self._image_brush_spin.value())
        self._update_image_controls_state()

        return tab

    def _build_pdf_tab(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(16)

        header = QtWidgets.QLabel("PDF mode (coming soon)", tab)
        header.setStyleSheet("font-size: 20px; font-weight: 600;")
        layout.addWidget(header)

        message = QtWidgets.QLabel(
            "Drop your design ideas here—PDF redaction will live in this workspace in the next milestone.",
            tab,
        )
        message.setWordWrap(True)
        message.setStyleSheet("color: #5c6a79; font-size: 14px;")
        layout.addWidget(message)
        layout.addStretch(1)
        return tab

    def _on_mode_changed(self, index: int) -> None:
        camera_active = index == self._camera_tab_index
        timer_ready = hasattr(self, "_timer")

        if camera_active:
            if timer_ready and getattr(self, "_auto_paused_stream", False) and not self._timer.isActive():
                self._start_streaming()
            self._auto_paused_stream = False
        else:
            if timer_ready and self._timer.isActive():
                self._auto_paused_stream = True
                self._stop_streaming()

        if self._toggle_stream_action is not None:
            self._toggle_stream_action.setEnabled(camera_active)
        if self._snapshot_action is not None:
            self._snapshot_action.setEnabled(camera_active and self._latest_frame is not None)

        image_active = index == self._image_tab_index
        if self._image_toolbar is not None:
            self._image_toolbar.setEnabled(image_active)
        self._sync_shortcut_states()

        if hasattr(self, "_status_label"):
            if camera_active:
                self._status_label.setText("Streaming from camera" if timer_ready and self._timer.isActive() else "Camera paused")
            elif image_active:
                if self._image_canvas and self._image_canvas.has_image():
                    self._status_label.setText("Image mode ready for refinements")
                else:
                    self._status_label.setText("Image mode: paste or open an image")
            else:
                self._status_label.setText("PDF mode under construction")

    def _register_shortcuts(self) -> None:
        self._paste_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.StandardKey.Paste, self)
        self._paste_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._paste_shortcut.activated.connect(self._handle_paste_request)

        self._open_image_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+O"), self)
        self._open_image_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._open_image_shortcut.activated.connect(self._on_image_open_clicked)

        self._save_image_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        self._save_image_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self._save_image_shortcut.activated.connect(self._on_image_save_clicked)

        self._sync_shortcut_states()

    def _sync_shortcut_states(self) -> None:
        if self._open_image_shortcut is not None:
            self._open_image_shortcut.setEnabled(True)
        save_enabled = False
        if self._tab_widget is not None and self._tab_widget.currentIndex() == self._image_tab_index:
            save_enabled = bool(self._image_canvas and self._image_canvas.has_image())
        if self._save_image_shortcut is not None:
            self._save_image_shortcut.setEnabled(save_enabled)

    def _handle_paste_request(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is None:
            return
        mime = clipboard.mimeData()
        if mime is None:
            self._status_label.setText("Clipboard is empty")
            return
        if mime.hasImage():
            image = clipboard.image()
            if image.isNull():
                self._status_label.setText("Clipboard image could not be read")
                return
            self._activate_image_mode()
            self._load_image_from_qimage(image, description="clipboard image")
            return
        if mime.hasUrls():
            extensions = {ext.lower() for ext in ImageCanvas._image_extensions()}
            for url in mime.urls():
                if not url.isLocalFile():
                    continue
                candidate = Path(url.toLocalFile())
                if candidate.suffix.lower() in extensions:
                    self._activate_image_mode()
                    self._load_image_from_path(candidate)
                    return
        self._status_label.setText("Clipboard does not contain image data")

    def _activate_image_mode(self) -> None:
        if self._tab_widget is None or self._image_tab_index < 0:
            return
        if self._tab_widget.currentIndex() != self._image_tab_index:
            self._tab_widget.setCurrentIndex(self._image_tab_index)
        if self._image_canvas is not None:
            self._image_canvas.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)

    def _initial_brush_color(self) -> QtGui.QColor:
        b, g, r = self.config.mask_color
        return QtGui.QColor(r, g, b)

    def _update_image_controls_state(self) -> None:
        has_image = bool(self._image_canvas and self._image_canvas.has_image())
        for button in (self._image_save_button, self._image_clear_button):
            if button is not None:
                button.setEnabled(has_image)
        if self._image_toolbar is not None:
            image_active = self._tab_widget is None or self._tab_widget.currentIndex() == self._image_tab_index
            self._image_toolbar.setEnabled(image_active)
        if self._image_hint_label is not None and not has_image:
            self._image_hint_label.setText(
                "Paste any image with Ctrl+V or drop files here to instantly anonymize faces and bodies."
            )
        self._sync_shortcut_states()
        if self._image_color_button is not None and self._image_canvas is not None:
            color = self._image_canvas.pen_color()
            luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
            text_color = "#0d1117" if luminance > 160 else "#f7fbff"
            self._image_color_button.setStyleSheet(
                (
                    "QPushButton {{ padding: 6px 12px; border-radius: 8px;"
                    f" background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
                    f" color: {text_color}; font-weight: 600; }}"
                )
            )

    def _on_image_tool_changed(self, _: int) -> None:
        if self._image_canvas is None or self._image_tool_combo is None:
            return
        data = self._image_tool_combo.currentData()
        if isinstance(data, DrawingTool):
            self._image_canvas.set_tool(data)

    def _on_image_brush_size_changed(self, value: int) -> None:
        if self._image_canvas is None:
            return
        self._image_canvas.set_pen_width(int(value))

    def _on_image_color_clicked(self) -> None:
        initial = self._initial_brush_color() if self._image_canvas is None else self._image_canvas.pen_color()
        color = QtWidgets.QColorDialog.getColor(initial, self, "Select brush color")
        if not color.isValid():
            return
        if self._image_canvas is not None:
            self._image_canvas.set_pen_color(color)
        self._update_image_controls_state()

    def _on_image_open_clicked(self) -> None:
        start_dir = str(self._image_last_directory) if self._image_last_directory else str(Path.home())
        filters = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tif *.tiff);;All files (*)"
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image", start_dir, filters)
        if not filename:
            return
        path = Path(filename)
        self._image_last_directory = path.parent
        self._activate_image_mode()
        self._load_image_from_path(path)

    def _on_image_save_clicked(self) -> None:
        if self._image_canvas is None or not self._image_canvas.has_image():
            QtWidgets.QMessageBox.information(self, "Export image", "Load an image before exporting.")
            return
        start_dir = str(self._image_last_directory) if self._image_last_directory else str(Path.home())
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        default_name = Path(start_dir) / f"face_gui-image-{timestamp}.png"
        filters = "PNG (*.png);;JPEG (*.jpg *.jpeg);;WEBP (*.webp);;All files (*)"
        filename, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export image",
            str(default_name),
            filters,
        )
        if not filename:
            return
        extension = Path(filename).suffix.lower()
        format_hint = "PNG"
        if "jpeg" in selected_filter.lower() or extension in {".jpg", ".jpeg"}:
            format_hint = "JPEG"
        elif "webp" in selected_filter.lower() or extension == ".webp":
            format_hint = "WEBP"

        image = self._image_canvas.composited_image()
        if image is None:
            QtWidgets.QMessageBox.warning(self, "Export image", "Unable to compose the image for export.")
            return
        if not image.save(filename, format_hint):
            QtWidgets.QMessageBox.warning(self, "Export image", "Saving failed. Check file permissions.")
            return
        self._image_last_directory = Path(filename).parent
        self._status_label.setText(f"Saved edited image to {Path(filename).name}")

    def _on_image_clear_clicked(self) -> None:
        if self._image_canvas is None:
            return
        if self._image_canvas.has_image():
            self._image_canvas.clear_overlay()
            self._status_label.setText("Cleared drawing overlay")
        else:
            self._image_canvas.clear()
            self._image_original_bgr = None
            self._image_segmentation_mask_buffer = None
            self._status_label.setText("Image workspace reset")
        self._update_image_controls_state()

    def _on_image_undo_clicked(self) -> None:
        if self._image_canvas is None:
            return
        self._image_canvas.undo()
        self._status_label.setText("Undo applied to overlay")

    def _on_image_dropped(self, paths: list[str]) -> None:
        for raw_path in paths:
            path = Path(raw_path)
            if path.exists():
                self._activate_image_mode()
                self._load_image_from_path(path)
                break

    def _load_image_from_path(self, path: Path) -> None:
        reader = QtGui.QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            QtWidgets.QMessageBox.warning(
                self,
                "Open image",
                f"Unable to read {path.name}.\n{reader.errorString()}",
            )
            return
        self._image_last_directory = path.parent
        self._load_image_from_qimage(image, description=path.name)

    def _load_image_from_qimage(self, image: QtGui.QImage, *, description: str) -> None:
        bgr = self._qimage_to_bgr(image)
        if bgr.size == 0:
            QtWidgets.QMessageBox.warning(self, "Image mode", "Unsupported image format.")
            return
        self._image_original_bgr = bgr
        self._image_segmentation_mask_buffer = None
        self._refresh_image_preview(preserve_overlay=False)
        height, width = bgr.shape[:2]
        message = f"Loaded {description} • {width}×{height}px"
        if self._image_hint_label is not None:
            self._image_hint_label.setText(
                f"{message}. Use the brush or rectangle to refine coverage where detection misses."
            )
        self._status_label.setText(f"Image mode: {message}")

    def _refresh_image_preview(self, preserve_overlay: bool = True) -> None:
        if self._image_canvas is None or self._image_original_bgr is None:
            self._update_image_controls_state()
            return
        frame = self._image_original_bgr.copy()
        processed = self._process_static_image(frame)
        qimage = self._bgr_to_qimage(processed)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self._image_canvas.set_base_pixmap(pixmap, reset_overlay=not preserve_overlay)
        if preserve_overlay:
            self._image_canvas.update()
        self._update_image_controls_state()

    def _process_static_image(self, frame: np.ndarray) -> np.ndarray:
        result = frame
        if self.config.enable_body_mask:
            result = self._annotate_body_with_segmenter(
                result,
                self._body_segmenter,
                "_image_segmentation_mask_buffer",
            )
        if self.config.enable_face_mask:
            result = self._annotate_faces_with_mesh(result, self._face_mesh)
        return result

    @staticmethod
    def _qimage_to_bgr(image: QtGui.QImage) -> np.ndarray:
        converted = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
        width = converted.width()
        height = converted.height()
        bytes_per_line = converted.bytesPerLine()
        buffer = converted.bits()
        buffer.setsize(bytes_per_line * height)
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, bytes_per_line))
        arr = arr[:, : width * 3].reshape((height, width, 3))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_qimage(frame: np.ndarray) -> QtGui.QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        bytes_per_line = channel * width
        image = QtGui.QImage(rgb.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        return image.copy()

    def _create_toolbar(self) -> None:
        toolbar = QtWidgets.QToolBar("Stream controls", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(QtCore.QSize(20, 20))
        toolbar.setObjectName("mainToolbar")

        pause_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause)
        toggle_action = QtWidgets.QAction(pause_icon, "Pause stream", self)
        toggle_action.setCheckable(True)
        toggle_action.setToolTip("Pause or resume the live feed without releasing the camera.")
        toggle_action.toggled.connect(self._on_stream_toggled)
        toolbar.addAction(toggle_action)
        self._toggle_stream_action = toggle_action

        snapshot_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton)
        snapshot_action = QtWidgets.QAction(snapshot_icon, "Save snapshot…", self)
        snapshot_action.setToolTip("Capture the current frame and save it as an image file.")
        snapshot_action.setEnabled(False)
        snapshot_action.triggered.connect(self._on_capture_snapshot)
        toolbar.addAction(snapshot_action)
        self._snapshot_action = snapshot_action

        toolbar.addSeparator()

        theme_icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarMenuButton)
        theme_action = QtWidgets.QAction(theme_icon, "Dark theme", self)
        theme_action.setCheckable(True)
        theme_action.setToolTip("Toggle a dark theme that is gentle on studio lighting.")
        theme_action.toggled.connect(self._on_theme_toggled)
        toolbar.addAction(theme_action)
        self._theme_action = theme_action

        toolbar.addSeparator()

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        help_label = QtWidgets.QLabel("Stream controls", self)
        help_label.setStyleSheet("color: #627387; font-size: 12px;")
        toolbar.addWidget(help_label)

        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, toolbar)

    def _create_control_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget(parent)
        panel.setObjectName("controlPanel")
        panel.setMinimumWidth(300)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 18, 18, 18)
        panel_layout.setSpacing(16)

        header = QtWidgets.QLabel("Live controls", panel)
        header.setObjectName("controlHeader")
        header.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        header.setStyleSheet("font-size: 18px; font-weight: 600;")
        panel_layout.addWidget(header)

        subtitle = QtWidgets.QLabel(
            "Fine-tune tracking quality, overlay appearance, and saved presets while the stream is running.",
            panel,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #5e6b78;")
        subtitle.setObjectName("controlSubtitle")
        panel_layout.addWidget(subtitle)

        panel_layout.addWidget(self._create_divider(panel))

        mask_group = QtWidgets.QGroupBox("Mask toggles", panel)
        mask_layout = QtWidgets.QVBoxLayout(mask_group)
        mask_layout.setSpacing(6)
        self._face_mask_checkbox = QtWidgets.QCheckBox("Face mask", mask_group)
        self._face_mask_checkbox.setChecked(self.config.enable_face_mask)
        self._face_mask_checkbox.toggled.connect(self._on_face_mask_toggled)
        self._face_mask_checkbox.setToolTip("Overlay an outline around detected faces.")
        mask_layout.addWidget(self._face_mask_checkbox)

        self._body_mask_checkbox = QtWidgets.QCheckBox("Body mask", mask_group)
        self._body_mask_checkbox.setChecked(self.config.enable_body_mask)
        self._body_mask_checkbox.toggled.connect(self._on_body_mask_toggled)
        self._body_mask_checkbox.setToolTip("Fill the segmented body area with the chosen color.")
        mask_layout.addWidget(self._body_mask_checkbox)
        mask_layout.addStretch(1)
        panel_layout.addWidget(mask_group)

        stream_group = QtWidgets.QGroupBox("Stream", panel)
        stream_form = QtWidgets.QFormLayout(stream_group)
        stream_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._fps_spin = QtWidgets.QSpinBox(stream_group)
        self._fps_spin.setRange(5, 120)
        self._fps_spin.setAccelerated(True)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setToolTip("Target refresh interval for the camera feed.")
        self._fps_spin.valueChanged.connect(self._on_target_fps_changed)
        stream_form.addRow("Target refresh", self._fps_spin)

        self._max_faces_spin = QtWidgets.QSpinBox(stream_group)
        self._max_faces_spin.setRange(1, 20)
        self._max_faces_spin.setAccelerated(True)
        self._max_faces_spin.setToolTip("Limit the number of faces passed to MediaPipe for detection.")
        self._max_faces_spin.valueChanged.connect(self._on_max_faces_changed)
        stream_form.addRow("Max faces", self._max_faces_spin)

        panel_layout.addWidget(stream_group)

        face_group = QtWidgets.QGroupBox("Face accuracy", panel)
        face_form = QtWidgets.QFormLayout(face_group)
        face_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._det_conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, face_group)
        self._det_conf_slider.setRange(0, 100)
        self._det_conf_slider.setSingleStep(1)
        self._det_conf_slider.setPageStep(5)
        self._det_conf_slider.setToolTip("Minimum confidence required before a new face is considered valid.")
        self._det_conf_slider.valueChanged.connect(self._on_detection_conf_changed)
        self._det_conf_value = QtWidgets.QLabel(face_group)
        self._det_conf_value.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        face_form.addRow("Detection conf", self._wrap_slider(self._det_conf_slider, self._det_conf_value))

        self._track_conf_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, face_group)
        self._track_conf_slider.setRange(0, 100)
        self._track_conf_slider.setSingleStep(1)
        self._track_conf_slider.setPageStep(5)
        self._track_conf_slider.setToolTip("Confidence required to keep tracking an already detected face.")
        self._track_conf_slider.valueChanged.connect(self._on_tracking_conf_changed)
        self._track_conf_value = QtWidgets.QLabel(face_group)
        self._track_conf_value.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        face_form.addRow("Tracking conf", self._wrap_slider(self._track_conf_slider, self._track_conf_value))

        panel_layout.addWidget(face_group)

        body_group = QtWidgets.QGroupBox("Body accuracy", panel)
        body_form = QtWidgets.QFormLayout(body_group)
        body_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._seg_threshold_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, body_group)
        self._seg_threshold_slider.setRange(0, 100)
        self._seg_threshold_slider.setSingleStep(1)
        self._seg_threshold_slider.setPageStep(5)
        self._seg_threshold_slider.setToolTip("Confidence cutoff before a pixel belongs to the foreground.")
        self._seg_threshold_slider.valueChanged.connect(self._on_segmentation_threshold_changed)
        self._seg_threshold_value = QtWidgets.QLabel(body_group)
        self._seg_threshold_value.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        body_form.addRow("Threshold", self._wrap_slider(self._seg_threshold_slider, self._seg_threshold_value))

        self._seg_smooth_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, body_group)
        self._seg_smooth_slider.setRange(0, 99)
        self._seg_smooth_slider.setSingleStep(1)
        self._seg_smooth_slider.setPageStep(5)
        self._seg_smooth_slider.setToolTip("Blend the mask over time to reduce flicker.")
        self._seg_smooth_slider.valueChanged.connect(self._on_segmentation_smooth_changed)
        self._seg_smooth_value = QtWidgets.QLabel(body_group)
        self._seg_smooth_value.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        body_form.addRow("Smooth factor", self._wrap_slider(self._seg_smooth_slider, self._seg_smooth_value))

        self._seg_kernel_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, body_group)
        self._seg_kernel_slider.setRange(1, _MAX_KERNEL_SIZE)
        self._seg_kernel_slider.setSingleStep(2)
        self._seg_kernel_slider.setPageStep(2)
        self._seg_kernel_slider.setToolTip("Spatial blur radius applied to the segmentation mask.")
        self._seg_kernel_slider.valueChanged.connect(self._on_segmentation_kernel_changed)
        self._seg_kernel_value = QtWidgets.QLabel(body_group)
        self._seg_kernel_value.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        body_form.addRow("Kernel size", self._wrap_slider(self._seg_kernel_slider, self._seg_kernel_value))

        self._seg_model_combo = QtWidgets.QComboBox(body_group)
        self._seg_model_combo.addItem("General (model 0)", 0)
        self._seg_model_combo.addItem("Landscape (model 1)", 1)
        self._seg_model_combo.setToolTip("Switch between MediaPipe's segmentation models.")
        self._seg_model_combo.currentIndexChanged.connect(self._on_segmentation_model_changed)
        body_form.addRow("Model", self._seg_model_combo)

        panel_layout.addWidget(body_group)

        appearance_group = QtWidgets.QGroupBox("Mask appearance", panel)
        appearance_form = QtWidgets.QFormLayout(appearance_group)
        appearance_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._mask_alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, appearance_group)
        self._mask_alpha_slider.setRange(0, 100)
        self._mask_alpha_slider.setSingleStep(1)
        self._mask_alpha_slider.setPageStep(5)
        self._mask_alpha_slider.setToolTip("Adjust the opacity of the overlay mask.")
        self._mask_alpha_slider.valueChanged.connect(self._on_mask_alpha_changed)
        self._mask_alpha_value = QtWidgets.QLabel(appearance_group)
        self._mask_alpha_value.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        appearance_form.addRow("Opacity", self._wrap_slider(self._mask_alpha_slider, self._mask_alpha_value))

        self._border_spin = QtWidgets.QSpinBox(appearance_group)
        self._border_spin.setRange(0, 20)
        self._border_spin.setAccelerated(True)
        self._border_spin.setToolTip("Thickness of the facial outline in pixels.")
        self._border_spin.valueChanged.connect(self._on_border_thickness_changed)
        appearance_form.addRow("Border width", self._border_spin)

        color_widget = QtWidgets.QWidget(appearance_group)
        color_layout = QtWidgets.QHBoxLayout(color_widget)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setSpacing(8)
        self._color_preview = QtWidgets.QLabel(color_widget)
        self._color_preview.setFixedSize(30, 20)
        self._color_preview.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self._color_preview.setLineWidth(1)
        self._color_preview.setToolTip("Current mask color in BGR order.")
        self._color_button = QtWidgets.QPushButton("Choose…", color_widget)
        self._color_button.clicked.connect(self._on_choose_mask_color)
        color_layout.addWidget(self._color_preview)
        color_layout.addWidget(self._color_button)
        color_layout.addStretch(1)
        appearance_form.addRow("Mask color", color_widget)

        panel_layout.addWidget(appearance_group)

        panel_layout.addWidget(self._create_divider(panel))

        button_row = QtWidgets.QHBoxLayout()
        self._import_button = QtWidgets.QPushButton("Import…", panel)
        self._import_button.setToolTip("Load settings from disk.")
        self._import_button.clicked.connect(self._on_import_settings)
        button_row.addWidget(self._import_button)

        self._export_button = QtWidgets.QPushButton("Export…", panel)
        self._export_button.setToolTip("Save the current configuration to disk.")
        self._export_button.clicked.connect(self._on_export_settings)
        button_row.addWidget(self._export_button)

        self._reset_button = QtWidgets.QPushButton("Reset", panel)
        self._reset_button.setToolTip("Revert to sensible defaults.")
        self._reset_button.clicked.connect(self._on_reset_settings)
        button_row.addWidget(self._reset_button)

        panel_layout.addLayout(button_row)

        helper_label = QtWidgets.QLabel(
            "Tip: export a tuned preset and re-import it later for consistent shoots.",
            panel,
        )
        helper_label.setWordWrap(True)
        helper_label.setStyleSheet("color: #556270; font-size: 12px;")
        panel_layout.addWidget(helper_label)

        panel_layout.addStretch(1)

        return panel

    def _wrap_slider(self, slider: QtWidgets.QSlider, label: QtWidgets.QLabel) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget(slider.parent())
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label.setMinimumWidth(48)
        layout.addWidget(slider, 1)
        layout.addWidget(label)
        return container

    def _create_divider(self, parent: QtWidgets.QWidget) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame(parent)
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setLineWidth(1)
        line.setStyleSheet("color: #1f2a33; margin-top: 6px; margin-bottom: 6px;")
        return line

    def _on_stream_toggled(self, paused: bool) -> None:
        if paused:
            self._stop_streaming()
        else:
            self._start_streaming()

    def _start_streaming(self) -> None:
        interval_ms = int(1000 / max(self.config.target_fps, 1))
        self._timer.setInterval(interval_ms)
        if not self._timer.isActive():
            self._timer.start()
        self._status_label.setText("Streaming from camera")
        self._fps_label.setText("… fps")
        self._last_frame_time = None
        if self._toggle_stream_action is not None:
            self._toggle_stream_action.blockSignals(True)
            self._toggle_stream_action.setChecked(False)
            self._toggle_stream_action.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause)
            )
            self._toggle_stream_action.setText("Pause stream")
            self._toggle_stream_action.setToolTip("Pause or resume the live feed without releasing the camera.")
            self._toggle_stream_action.blockSignals(False)

    def _stop_streaming(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        self._status_label.setText("Stream paused")
        self._fps_label.setText("– fps")
        if self._toggle_stream_action is not None:
            self._toggle_stream_action.blockSignals(True)
            self._toggle_stream_action.setChecked(True)
            self._toggle_stream_action.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
            )
            self._toggle_stream_action.setText("Resume stream")
            self._toggle_stream_action.setToolTip("Resume processing frames from the camera.")
            self._toggle_stream_action.blockSignals(False)

    def _reconfigure_timer(self) -> None:
        if not hasattr(self, "_timer"):
            return
        interval_ms = int(1000 / max(self.config.target_fps, 1))
        self._timer.setInterval(interval_ms)
        if self._timer.isActive():
            self._status_label.setText(
                f"Streaming from camera (target {int(self.config.target_fps)} fps)"
            )

    def _on_capture_snapshot(self) -> None:
        if self._latest_frame is None:
            QtWidgets.QMessageBox.information(
                self,
                "Save Snapshot",
                "No frame has been captured yet. Wait for the stream to start before saving.",
            )
            return

        pictures_dir = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.PicturesLocation
        )
        if not pictures_dir:
            pictures_dir = str(Path.home())
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        suggested = Path(pictures_dir) / f"face_gui-{timestamp}.png"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Snapshot",
            str(suggested),
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)",
        )
        if not filename:
            return

        rgb_frame = self._latest_frame
        assert rgb_frame is not None
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        try:
            success = cv2.imwrite(filename, bgr_frame)
        except Exception:  # pragma: no cover - filesystem pathologies
            success = False
        if not success:
            QtWidgets.QMessageBox.warning(self, "Save Snapshot", "Unable to save the snapshot to disk.")
            return

        self._status_label.setText(f"Snapshot saved to {Path(filename).name}")

    def _on_theme_toggled(self, enabled: bool) -> None:
        self._dark_theme_enabled = enabled
        self._apply_theme_palette(enabled)
        self._status_label.setText("Dark theme enabled" if enabled else "Dark theme disabled")

    def _apply_theme_palette(self, enabled: bool) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        if not enabled:
            app.setPalette(app.style().standardPalette())
            app.setStyleSheet("")
            return

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(18, 23, 29))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(220, 230, 240))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(13, 18, 24))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(16, 22, 28))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(53, 53, 53))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(238, 238, 238))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(220, 230, 240))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(24, 31, 38))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(220, 230, 240))
        palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 0, 0))
        palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(45, 140, 240))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(18, 23, 29))
        app.setPalette(palette)
        app.setStyleSheet(
            "QToolTip { color: #f0f0f0; background-color: #353535; border: 1px solid #3c3c3c; }"
        )

    def _on_target_fps_changed(self, value: int) -> None:
        if not hasattr(self, "_fps_spin"):
            return
        minimum = self._fps_spin.minimum()
        maximum = self._fps_spin.maximum()
        clamped = int(np.clip(value, minimum, maximum))
        if clamped != value:
            self._fps_spin.blockSignals(True)
            self._fps_spin.setValue(clamped)
            self._fps_spin.blockSignals(False)
        if clamped == self.config.target_fps:
            return
        self.config.target_fps = clamped
        self._reconfigure_timer()
        self._status_label.setText(f"Target FPS set to {clamped}")
        self._persist_settings()

    def _on_max_faces_changed(self, value: int) -> None:
        minimum = self._max_faces_spin.minimum()
        maximum = self._max_faces_spin.maximum()
        clamped = int(np.clip(value, minimum, maximum))
        if clamped != value:
            self._max_faces_spin.blockSignals(True)
            self._max_faces_spin.setValue(clamped)
            self._max_faces_spin.blockSignals(False)
        if clamped == self.config.max_faces:
            return

        previous = self.config.max_faces
        self.config.max_faces = clamped
        try:
            self._rebuild_face_mesh()
        except RuntimeError as exc:
            self.config.max_faces = previous
            self._max_faces_spin.blockSignals(True)
            self._max_faces_spin.setValue(previous)
            self._max_faces_spin.blockSignals(False)
            try:
                self._rebuild_face_mesh()
            except RuntimeError:
                pass
            self._status_label.setText(str(exc))
            return

        self._status_label.setText(f"Detecting up to {clamped} faces")
        self._persist_settings()
        self._refresh_image_preview()

    def _on_mask_alpha_changed(self, slider_value: int) -> None:
        slider_value = int(
            np.clip(slider_value, self._mask_alpha_slider.minimum(), self._mask_alpha_slider.maximum())
        )
        value = slider_value / 100.0
        self._mask_alpha_value.setText(f"{value:.2f}")
        if abs(value - self.config.mask_alpha) < 1e-6:
            return
        self.config.mask_alpha = value
        self._status_label.setText("Updated mask opacity")
        self._persist_settings()
        self._refresh_image_preview()

    def _on_border_thickness_changed(self, value: int) -> None:
        minimum = self._border_spin.minimum()
        maximum = self._border_spin.maximum()
        clamped = int(np.clip(value, minimum, maximum))
        if clamped != value:
            self._border_spin.blockSignals(True)
            self._border_spin.setValue(clamped)
            self._border_spin.blockSignals(False)
        if clamped == self.config.border_thickness:
            return
        self.config.border_thickness = clamped
        self._status_label.setText("Updated border width")
        self._persist_settings()
        self._refresh_image_preview()

    def _on_choose_mask_color(self) -> None:
        b, g, r = self.config.mask_color
        initial = QtGui.QColor(r, g, b)
        color = QtWidgets.QColorDialog.getColor(initial, self, "Select Mask Color")
        if not color.isValid():
            return
        new_color = (color.blue(), color.green(), color.red())
        if new_color == self.config.mask_color:
            return
        self.config.mask_color = new_color
        self._update_color_preview()
        self._update_image_controls_state()
        self._status_label.setText("Updated mask color")
        self._persist_settings()
        self._refresh_image_preview()

    def _update_color_preview(self) -> None:
        if not hasattr(self, "_color_preview") or self._color_preview is None:
            return
        b, g, r = self.config.mask_color
        self._color_preview.setStyleSheet(
            f"border: 1px solid #2a2f36; border-radius: 4px; background-color: rgb({r}, {g}, {b});"
        )
        self._color_preview.setToolTip(f"Mask color (BGR): {b}, {g}, {r}")

    def _sync_controls_from_config(self) -> None:
        if not hasattr(self, "_face_mask_checkbox"):
            return

        self._face_mask_checkbox.blockSignals(True)
        self._face_mask_checkbox.setChecked(self.config.enable_face_mask)
        self._face_mask_checkbox.blockSignals(False)

        self._body_mask_checkbox.blockSignals(True)
        self._body_mask_checkbox.setChecked(self.config.enable_body_mask)
        self._body_mask_checkbox.blockSignals(False)

        self.config.detection_confidence = self._set_slider_float(
            self._det_conf_slider,
            self._det_conf_value,
            self.config.detection_confidence,
            scale=100,
            decimals=2,
            min_value=0.0,
            max_value=1.0,
        )

        self.config.tracking_confidence = self._set_slider_float(
            self._track_conf_slider,
            self._track_conf_value,
            self.config.tracking_confidence,
            scale=100,
            decimals=2,
            min_value=0.0,
            max_value=1.0,
        )

        self.config.segmentation_threshold = self._set_slider_float(
            self._seg_threshold_slider,
            self._seg_threshold_value,
            self.config.segmentation_threshold,
            scale=100,
            decimals=2,
            min_value=0.0,
            max_value=1.0,
        )

        self.config.segmentation_smooth_factor = self._set_slider_float(
            self._seg_smooth_slider,
            self._seg_smooth_value,
            self.config.segmentation_smooth_factor,
            scale=100,
            decimals=2,
            min_value=0.0,
            max_value=0.99,
        )

        self.config.segmentation_kernel_size = self._set_kernel_slider_value(self.config.segmentation_kernel_size)

        model_index = self._seg_model_combo.findData(int(np.clip(self.config.segmentation_model_selection, 0, 1)))
        self._seg_model_combo.blockSignals(True)
        if model_index >= 0:
            self._seg_model_combo.setCurrentIndex(model_index)
        else:
            self._seg_model_combo.setCurrentIndex(0)
            data = self._seg_model_combo.currentData()
            if data is not None:
                self.config.segmentation_model_selection = int(data)
        self._seg_model_combo.blockSignals(False)

        if hasattr(self, "_fps_spin") and self._fps_spin is not None:
            minimum = self._fps_spin.minimum()
            maximum = self._fps_spin.maximum()
            fps_value = int(np.clip(self.config.target_fps, minimum, maximum))
            self._fps_spin.blockSignals(True)
            self._fps_spin.setValue(fps_value)
            self._fps_spin.blockSignals(False)
            self.config.target_fps = fps_value

        if hasattr(self, "_max_faces_spin") and self._max_faces_spin is not None:
            minimum = self._max_faces_spin.minimum()
            maximum = self._max_faces_spin.maximum()
            faces_value = int(np.clip(self.config.max_faces, minimum, maximum))
            self._max_faces_spin.blockSignals(True)
            self._max_faces_spin.setValue(faces_value)
            self._max_faces_spin.blockSignals(False)
            self.config.max_faces = faces_value

        if hasattr(self, "_mask_alpha_slider") and self._mask_alpha_slider is not None:
            self.config.mask_alpha = self._set_slider_float(
                self._mask_alpha_slider,
                self._mask_alpha_value,
                self.config.mask_alpha,
                scale=100,
                decimals=2,
                min_value=0.0,
                max_value=1.0,
            )

        if hasattr(self, "_border_spin") and self._border_spin is not None:
            minimum = self._border_spin.minimum()
            maximum = self._border_spin.maximum()
            border_value = int(np.clip(self.config.border_thickness, minimum, maximum))
            self._border_spin.blockSignals(True)
            self._border_spin.setValue(border_value)
            self._border_spin.blockSignals(False)
            self.config.border_thickness = border_value

        self._update_color_preview()

        self._update_control_states()

    def _set_slider_float(
        self,
        slider: QtWidgets.QSlider,
        label: QtWidgets.QLabel,
        value: float,
        *,
        scale: int,
        decimals: int,
        min_value: float,
        max_value: float,
    ) -> float:
        clamped = float(np.clip(value, min_value, max_value))
        slider_min = slider.minimum()
        slider_max = slider.maximum()
        slider_value = int(round(clamped * scale))
        slider_value = max(slider_min, min(slider_value, slider_max))
        slider.blockSignals(True)
        slider.setValue(slider_value)
        slider.blockSignals(False)
        effective = slider_value / scale
        label.setText(f"{effective:.{decimals}f}")
        return effective

    def _set_kernel_slider_value(self, value: int) -> int:
        minimum = self._seg_kernel_slider.minimum()
        maximum = self._seg_kernel_slider.maximum()
        adjusted = self._ensure_odd(max(minimum, min(int(value), maximum)))
        self._seg_kernel_slider.blockSignals(True)
        self._seg_kernel_slider.setValue(adjusted)
        self._seg_kernel_slider.blockSignals(False)
        self._seg_kernel_value.setText(str(adjusted))
        return adjusted

    @staticmethod
    def _ensure_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    def _update_control_states(self) -> None:
        face_enabled = self.config.enable_face_mask
        for widget in (self._det_conf_slider, self._det_conf_value, self._track_conf_slider, self._track_conf_value):
            widget.setEnabled(face_enabled)
        if hasattr(self, "_max_faces_spin") and self._max_faces_spin is not None:
            self._max_faces_spin.setEnabled(face_enabled)

        body_enabled = self.config.enable_body_mask
        for widget in (
            self._seg_threshold_slider,
            self._seg_threshold_value,
            self._seg_smooth_slider,
            self._seg_smooth_value,
            self._seg_kernel_slider,
            self._seg_kernel_value,
            self._seg_model_combo,
        ):
            widget.setEnabled(body_enabled)

        mask_enabled = self.config.enable_face_mask or self.config.enable_body_mask
        extra_widgets = [
            getattr(self, "_mask_alpha_slider", None),
            getattr(self, "_mask_alpha_value", None),
            getattr(self, "_border_spin", None),
            getattr(self, "_color_button", None),
            getattr(self, "_color_preview", None),
        ]
        for widget in extra_widgets:
            if widget is not None:
                widget.setEnabled(mask_enabled)

    def _reset_segmentation_buffer(self) -> None:
        self._segmentation_mask_buffer = None

    def _reset_image_segmentation_buffer(self) -> None:
        self._image_segmentation_mask_buffer = None

    def _rebuild_face_mesh(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

        if not self.config.enable_face_mask:
            return

        max_faces = max(int(self.config.max_faces), 1)
        try:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=float(np.clip(self.config.detection_confidence, 0.0, 1.0)),
                min_tracking_confidence=float(np.clip(self.config.tracking_confidence, 0.0, 1.0)),
            )
        except Exception as exc:  # pragma: no cover - initialization failure
            self._face_mesh = None
            raise RuntimeError("Unable to initialize MediaPipe Face Mesh.") from exc

    def _rebuild_body_segmenter(self) -> None:
        if self._body_segmenter is not None:
            self._body_segmenter.close()
            self._body_segmenter = None

        self._reset_segmentation_buffer()
        self._reset_image_segmentation_buffer()

        if not self.config.enable_body_mask:
            return

        try:
            self._body_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=int(np.clip(self.config.segmentation_model_selection, 0, 1))
            )
        except Exception as exc:  # pragma: no cover - initialization failure
            self._body_segmenter = None
            raise RuntimeError("Unable to initialize MediaPipe Selfie Segmentation.") from exc

    def _on_detection_conf_changed(self, slider_value: int) -> None:
        slider_value = int(np.clip(slider_value, self._det_conf_slider.minimum(), self._det_conf_slider.maximum()))
        value = slider_value / 100.0
        self._det_conf_value.setText(f"{value:.2f}")
        if abs(value - self.config.detection_confidence) < 1e-6:
            return

        previous = self.config.detection_confidence
        self.config.detection_confidence = value
        try:
            self._rebuild_face_mesh()
        except RuntimeError as exc:
            self.config.detection_confidence = previous
            self._det_conf_slider.blockSignals(True)
            self._det_conf_slider.setValue(int(round(previous * 100)))
            self._det_conf_slider.blockSignals(False)
            self._det_conf_value.setText(f"{previous:.2f}")
            try:
                self._rebuild_face_mesh()
            except RuntimeError:
                pass
            self._status_label.setText(str(exc))
            return
        self._status_label.setText("Updated face detection confidence")
        self._persist_settings()
        self._refresh_image_preview()

    def _on_tracking_conf_changed(self, slider_value: int) -> None:
        slider_value = int(np.clip(slider_value, self._track_conf_slider.minimum(), self._track_conf_slider.maximum()))
        value = slider_value / 100.0
        self._track_conf_value.setText(f"{value:.2f}")
        if abs(value - self.config.tracking_confidence) < 1e-6:
            return

        previous = self.config.tracking_confidence
        self.config.tracking_confidence = value
        try:
            self._rebuild_face_mesh()
        except RuntimeError as exc:
            self.config.tracking_confidence = previous
            self._track_conf_slider.blockSignals(True)
            self._track_conf_slider.setValue(int(round(previous * 100)))
            self._track_conf_slider.blockSignals(False)
            self._track_conf_value.setText(f"{previous:.2f}")
            try:
                self._rebuild_face_mesh()
            except RuntimeError:
                pass
            self._status_label.setText(str(exc))
            return
        self._status_label.setText("Updated face tracking confidence")
        self._persist_settings()
        self._refresh_image_preview()

    def _on_segmentation_threshold_changed(self, slider_value: int) -> None:
        slider_value = int(
            np.clip(slider_value, self._seg_threshold_slider.minimum(), self._seg_threshold_slider.maximum())
        )
        value = slider_value / 100.0
        self._seg_threshold_value.setText(f"{value:.2f}")
        if abs(value - self.config.segmentation_threshold) < 1e-6:
            return

        self.config.segmentation_threshold = value
        self._status_label.setText("Updated body threshold")
        self._persist_settings()
        self._refresh_image_preview()

    def _on_segmentation_smooth_changed(self, slider_value: int) -> None:
        slider_value = int(np.clip(slider_value, self._seg_smooth_slider.minimum(), self._seg_smooth_slider.maximum()))
        value = slider_value / 100.0
        self._seg_smooth_value.setText(f"{value:.2f}")
        if abs(value - self.config.segmentation_smooth_factor) < 1e-6:
            return

        self.config.segmentation_smooth_factor = value
        self._reset_segmentation_buffer()
        self._reset_image_segmentation_buffer()
        self._status_label.setText("Updated body smoothing")
        self._persist_settings()
        self._refresh_image_preview()

    def _on_segmentation_kernel_changed(self, raw_value: int) -> None:
        minimum = self._seg_kernel_slider.minimum()
        maximum = self._seg_kernel_slider.maximum()
        adjusted = self._ensure_odd(max(minimum, min(int(raw_value), maximum)))
        if adjusted != raw_value:
            self._seg_kernel_slider.blockSignals(True)
            self._seg_kernel_slider.setValue(adjusted)
            self._seg_kernel_slider.blockSignals(False)
        self._seg_kernel_value.setText(str(adjusted))
        if adjusted == self.config.segmentation_kernel_size:
            return

        self.config.segmentation_kernel_size = adjusted
        self._reset_segmentation_buffer()
        self._reset_image_segmentation_buffer()
        self._status_label.setText("Updated body kernel size")
        self._persist_settings()
        self._refresh_image_preview()

    def _on_segmentation_model_changed(self, index: int) -> None:
        data = self._seg_model_combo.itemData(index)
        if data is None:
            return

        selection = int(data)
        if selection == self.config.segmentation_model_selection:
            return

        previous = self.config.segmentation_model_selection
        self.config.segmentation_model_selection = selection
        try:
            self._rebuild_body_segmenter()
        except RuntimeError as exc:
            self.config.segmentation_model_selection = previous
            self._seg_model_combo.blockSignals(True)
            prev_index = self._seg_model_combo.findData(previous)
            if prev_index >= 0:
                self._seg_model_combo.setCurrentIndex(prev_index)
            self._seg_model_combo.blockSignals(False)
            try:
                self._rebuild_body_segmenter()
            except RuntimeError:
                pass
            self._status_label.setText(str(exc))
            return
        self._status_label.setText("Updated body segmentation model")
        self._persist_settings()
        self._refresh_image_preview()

    def _on_face_mask_toggled(self, checked: bool) -> None:
        if checked == self.config.enable_face_mask:
            return

        if not checked:
            self.config.enable_face_mask = False
            if self._face_mesh is not None:
                self._face_mesh.close()
                self._face_mesh = None
            self._update_control_states()
            self._status_label.setText("Face mask disabled")
            self._persist_settings()
            self._refresh_image_preview()
            return

        self.config.enable_face_mask = True
        try:
            self._rebuild_face_mesh()
        except RuntimeError as exc:
            self.config.enable_face_mask = False
            self._face_mask_checkbox.blockSignals(True)
            self._face_mask_checkbox.setChecked(False)
            self._face_mask_checkbox.blockSignals(False)
            self._update_control_states()
            try:
                self._rebuild_face_mesh()
            except RuntimeError:
                pass
            self._status_label.setText(str(exc))
            return
        self._update_control_states()
        self._status_label.setText("Face mask enabled")
        self._persist_settings()
        self._refresh_image_preview()

    def _on_body_mask_toggled(self, checked: bool) -> None:
        if checked == self.config.enable_body_mask:
            return

        if not checked:
            self.config.enable_body_mask = False
            if self._body_segmenter is not None:
                self._body_segmenter.close()
                self._body_segmenter = None
            self._reset_segmentation_buffer()
            self._reset_image_segmentation_buffer()
            self._update_control_states()
            self._status_label.setText("Body mask disabled")
            self._persist_settings()
            self._refresh_image_preview()
            return

        self.config.enable_body_mask = True
        try:
            self._rebuild_body_segmenter()
        except RuntimeError as exc:
            self.config.enable_body_mask = False
            self._body_mask_checkbox.blockSignals(True)
            self._body_mask_checkbox.setChecked(False)
            self._body_mask_checkbox.blockSignals(False)
            self._update_control_states()
            try:
                self._rebuild_body_segmenter()
            except RuntimeError:
                pass
            self._status_label.setText(str(exc))
            return
        self._update_control_states()
        self._status_label.setText("Body mask enabled")
        self._persist_settings()
        self._refresh_image_preview()

    def _current_settings_payload(self) -> dict[str, Any]:
        return {field: getattr(self.config, field) for field in _PERSISTED_FIELDS}

    def _resolve_settings_path(self) -> Optional[Path]:
        locations: list[Path] = []
        standard = QtCore.QStandardPaths.writableLocation(
            QtCore.QStandardPaths.StandardLocation.AppConfigLocation
        )
        if standard:
            locations.append(Path(standard))
        locations.append(Path.home() / ".face_gui")
        for base in locations:
            try:
                base.mkdir(parents=True, exist_ok=True)
            except Exception:
                continue
            return base / "settings.json"
        return None

    def _load_persisted_settings(self) -> None:
        path = self._resolve_settings_path()
        self._settings_file = path
        if path is None or not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return
        if not isinstance(data, Mapping):
            return
        self._update_config_from_mapping(data)

    def _persist_settings(self) -> None:
        path = self._settings_file or self._resolve_settings_path()
        if path is None:
            return
        self._settings_file = path
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(self._current_settings_payload(), handle, indent=2)
        except Exception:
            pass

    def _update_config_from_mapping(self, data: Mapping[str, Any]) -> tuple[bool, bool, bool, bool]:
        face_rebuild = False
        body_rebuild = False
        reset_buffer = False
        fps_changed = False

        face_mask = self._coerce_bool(data.get("enable_face_mask"))
        if face_mask is not None and face_mask != self.config.enable_face_mask:
            self.config.enable_face_mask = face_mask
            face_rebuild = True

        body_mask = self._coerce_bool(data.get("enable_body_mask"))
        if body_mask is not None and body_mask != self.config.enable_body_mask:
            self.config.enable_body_mask = body_mask
            body_rebuild = True

        det_conf = self._coerce_float(data.get("detection_confidence"))
        if det_conf is not None:
            det_conf = float(np.clip(det_conf, 0.0, 1.0))
            if abs(det_conf - self.config.detection_confidence) >= 1e-6:
                self.config.detection_confidence = det_conf
                face_rebuild = True

        track_conf = self._coerce_float(data.get("tracking_confidence"))
        if track_conf is not None:
            track_conf = float(np.clip(track_conf, 0.0, 1.0))
            if abs(track_conf - self.config.tracking_confidence) >= 1e-6:
                self.config.tracking_confidence = track_conf
                face_rebuild = True

        seg_threshold = self._coerce_float(data.get("segmentation_threshold"))
        if seg_threshold is not None:
            seg_threshold = float(np.clip(seg_threshold, 0.0, 1.0))
            if abs(seg_threshold - self.config.segmentation_threshold) >= 1e-6:
                self.config.segmentation_threshold = seg_threshold

        seg_smooth = self._coerce_float(data.get("segmentation_smooth_factor"))
        if seg_smooth is not None:
            seg_smooth = float(np.clip(seg_smooth, 0.0, 0.99))
            if abs(seg_smooth - self.config.segmentation_smooth_factor) >= 1e-6:
                self.config.segmentation_smooth_factor = seg_smooth
                reset_buffer = True

        seg_kernel = self._coerce_int(data.get("segmentation_kernel_size"))
        if seg_kernel is not None:
            seg_kernel = self._ensure_odd(max(1, min(seg_kernel, _MAX_KERNEL_SIZE)))
            if seg_kernel != self.config.segmentation_kernel_size:
                self.config.segmentation_kernel_size = seg_kernel
                reset_buffer = True

        seg_model = self._coerce_int(data.get("segmentation_model_selection"))
        if seg_model is not None:
            seg_model = int(np.clip(seg_model, 0, 1))
            if seg_model != self.config.segmentation_model_selection:
                self.config.segmentation_model_selection = seg_model
                body_rebuild = True

        mask_alpha = self._coerce_float(data.get("mask_alpha"))
        if mask_alpha is not None:
            mask_alpha = float(np.clip(mask_alpha, 0.0, 1.0))
            if abs(mask_alpha - self.config.mask_alpha) >= 1e-6:
                self.config.mask_alpha = mask_alpha

        mask_color = self._coerce_color(data.get("mask_color"))
        if mask_color is not None and mask_color != self.config.mask_color:
            self.config.mask_color = mask_color

        border_thickness = self._coerce_int(data.get("border_thickness"))
        if border_thickness is not None:
            border_thickness = max(0, border_thickness)
            if border_thickness != self.config.border_thickness:
                self.config.border_thickness = border_thickness

        max_faces = self._coerce_int(data.get("max_faces"))
        if max_faces is not None:
            max_faces = max(1, max_faces)
            if max_faces != self.config.max_faces:
                self.config.max_faces = max_faces
                face_rebuild = True

        target_fps = self._coerce_int(data.get("target_fps"))
        if target_fps is not None:
            target_fps = max(1, target_fps)
            if target_fps != self.config.target_fps:
                self.config.target_fps = target_fps
                fps_changed = True

        return face_rebuild, body_rebuild, reset_buffer, fps_changed

    def _on_import_settings(self) -> None:
        start_dir = str(self._settings_file.parent) if self._settings_file else str(Path.home())
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Settings",
            start_dir,
            "JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        try:
            with Path(filename).open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import Settings", f"Unable to read settings:\n{exc}")
            return

        if not isinstance(data, Mapping):
            QtWidgets.QMessageBox.warning(
                self,
                "Import Settings",
                "Settings file must contain a JSON object.",
            )
            return

        previous = asdict(self.config)
        face_changed, body_changed, reset_buffer, fps_changed = self._update_config_from_mapping(data)

        try:
            if face_changed:
                self._rebuild_face_mesh()
            if body_changed:
                self._rebuild_body_segmenter()
            elif reset_buffer:
                self._reset_segmentation_buffer()
                self._reset_image_segmentation_buffer()
            if fps_changed:
                self._reconfigure_timer()
        except RuntimeError as exc:
            revert_face, revert_body, revert_reset, revert_fps = self._update_config_from_mapping(previous)
            try:
                if revert_face:
                    self._rebuild_face_mesh()
            except RuntimeError:
                pass
            try:
                if revert_body:
                    self._rebuild_body_segmenter()
            except RuntimeError:
                pass
            if revert_reset and not revert_body:
                self._reset_segmentation_buffer()
                self._reset_image_segmentation_buffer()
            if revert_fps:
                self._reconfigure_timer()
            self._sync_controls_from_config()
            QtWidgets.QMessageBox.warning(self, "Import Settings", str(exc))
            self._status_label.setText(str(exc))
            return

        self._sync_controls_from_config()
        self._persist_settings()
        if fps_changed:
            self._reconfigure_timer()
        self._status_label.setText(f"Imported settings from {Path(filename).name}")
        self._refresh_image_preview()

    def _on_export_settings(self) -> None:
        start_dir = str(self._settings_file.parent) if self._settings_file else str(Path.home())
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Settings",
            start_dir,
            "JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        payload = self._current_settings_payload()
        try:
            with Path(filename).open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export Settings", f"Unable to write settings:\n{exc}")
            return

        self._status_label.setText(f"Exported settings to {Path(filename).name}")

    def _on_reset_settings(self) -> None:
        defaults = DetectionConfig()
        previous = asdict(self.config)
        reset_values = {field: getattr(defaults, field) for field in _PERSISTED_FIELDS}
        face_changed, body_changed, reset_buffer, fps_changed = self._update_config_from_mapping(reset_values)

        try:
            if face_changed:
                self._rebuild_face_mesh()
            if body_changed:
                self._rebuild_body_segmenter()
            elif reset_buffer:
                self._reset_segmentation_buffer()
                self._reset_image_segmentation_buffer()
            if fps_changed:
                self._reconfigure_timer()
        except RuntimeError as exc:
            revert_face, revert_body, revert_reset, revert_fps = self._update_config_from_mapping(previous)
            try:
                if revert_face:
                    self._rebuild_face_mesh()
            except RuntimeError:
                pass
            try:
                if revert_body:
                    self._rebuild_body_segmenter()
            except RuntimeError:
                pass
            if revert_reset and not revert_body:
                self._reset_segmentation_buffer()
                self._reset_image_segmentation_buffer()
            if revert_fps:
                self._reconfigure_timer()
            self._sync_controls_from_config()
            QtWidgets.QMessageBox.warning(self, "Reset Settings", str(exc))
            self._status_label.setText(str(exc))
            return

        self._sync_controls_from_config()
        self._persist_settings()
        if fps_changed:
            self._reconfigure_timer()
        self._status_label.setText("Settings reset to defaults")
        self._refresh_image_preview()

    @staticmethod
    def _coerce_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_color(value: Any) -> Optional[Tuple[int, int, int]]:
        def _normalize(components: Iterable[Any]) -> Optional[Tuple[int, int, int]]:
            values: list[int] = []
            for component in components:
                try:
                    values.append(int(float(component)))
                except (TypeError, ValueError):
                    return None
            if len(values) != 3:
                return None
            if any(not 0 <= part <= 255 for part in values):
                return None
            b, g, r = values
            return (b, g, r)

        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return _normalize(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("#") and len(stripped) in {7, 9}:
                try:
                    r = int(stripped[1:3], 16)
                    g = int(stripped[3:5], 16)
                    b = int(stripped[5:7], 16)
                except ValueError:
                    return None
                return (b, g, r)
            if "," in stripped:
                return _normalize(part.strip() for part in stripped.split(","))
        return None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt overrides)
        self._persist_settings()
        self._timer.stop()
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        if getattr(self, "_body_segmenter", None) is not None:
            self._body_segmenter.close()
            self._body_segmenter = None
        self._segmentation_mask_buffer = None
        self._image_segmentation_mask_buffer = None
        if self._camera.isOpened():
            self._camera.release()
        super().closeEvent(event)

    def _update_frame(self) -> None:
        ok, frame = self._camera.read()
        if not ok or frame is None:
            self._status_label.setText("Waiting for camera…")
            self._video_label.setPixmap(QtGui.QPixmap())
            self._video_label.setText("Camera not ready\nCheck that the selected device is available.")
            self._fps_label.setText("– fps")
            self._latest_frame = None
            return

        annotated = self._apply_masks(frame)
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self._latest_frame = rgb_frame.copy()
        height, width, channel_count = rgb_frame.shape
        bytes_per_line = channel_count * width

        image = QtGui.QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        )

        # Make a deep copy so the underlying numpy buffer can be released safely.
        pixmap = QtGui.QPixmap.fromImage(image.copy())
        target_size = self._video_label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            self._video_label.setPixmap(pixmap)
        else:
            scaled = pixmap.scaled(
                target_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self._video_label.setPixmap(scaled)
        self._video_label.setText("")
        self._status_label.setText("Streaming from camera")
        if self._snapshot_action is not None and not self._snapshot_action.isEnabled():
            self._snapshot_action.setEnabled(True)

        now = time.perf_counter()
        if self._last_frame_time is not None:
            delta = max(now - self._last_frame_time, 1e-6)
            fps = 1.0 / delta
            self._fps_label.setText(f"{fps:4.1f} fps")
        else:
            self._fps_label.setText("… fps")
        self._last_frame_time = now

    def _apply_masks(self, frame: np.ndarray) -> np.ndarray:
        if self.config.enable_body_mask:
            frame = self._annotate_body(frame)
        if self.config.enable_face_mask:
            frame = self._annotate_faces(frame)
        return frame

    def _annotate_faces(self, frame: np.ndarray) -> np.ndarray:
        return self._annotate_faces_with_mesh(frame, self._face_mesh)

    def _annotate_body(self, frame: np.ndarray) -> np.ndarray:
        return self._annotate_body_with_segmenter(frame, self._body_segmenter, "_segmentation_mask_buffer")

    def _annotate_faces_with_mesh(
        self,
        frame: np.ndarray,
        mesh: mp.solutions.face_mesh.FaceMesh | None,
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return frame
        if mesh is None:
            return frame

        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb_frame)
        frame.flags.writeable = True

        if not getattr(results, "multi_face_landmarks", None):
            return frame

        color = self.config.mask_color
        alpha = float(np.clip(self.config.mask_alpha, 0.0, 1.0))
        border = max(self.config.border_thickness, 0)

        polygons = list(self._generate_face_polygons(results.multi_face_landmarks, width, height))
        if not polygons:
            return frame

        if alpha >= 0.999:
            for polygon in polygons:
                cv2.fillPoly(frame, [polygon], color)
                if border > 0:
                    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=border)
            return frame

        overlay = frame.copy()
        for polygon in polygons:
            cv2.fillPoly(overlay, [polygon], color)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

        if border > 0:
            for polygon in polygons:
                cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=border)

        return frame

    def _annotate_body_with_segmenter(
        self,
        frame: np.ndarray,
        segmenter: mp.solutions.selfie_segmentation.SelfieSegmentation | None,
        buffer_attribute: str,
    ) -> np.ndarray:
        if segmenter is None:
            return frame

        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return frame

        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmenter.process(rgb_frame)
        frame.flags.writeable = True

        segmentation_mask: Optional[np.ndarray] = getattr(results, "segmentation_mask", None)
        if segmentation_mask is None:
            setattr(self, buffer_attribute, None)
            return frame

        mask_prob = np.asarray(segmentation_mask, dtype=np.float32)
        buffer_value: Optional[np.ndarray] = getattr(self, buffer_attribute, None)
        smooth_factor = float(np.clip(self.config.segmentation_smooth_factor, 0.0, 0.99))
        if smooth_factor > 0 and buffer_value is not None:
            mask_prob = smooth_factor * buffer_value + (1.0 - smooth_factor) * mask_prob

        kernel_size = max(int(self.config.segmentation_kernel_size), 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size >= 3:
            mask_prob = cv2.GaussianBlur(mask_prob, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)
        mask_prob = np.clip(mask_prob, 0.0, 1.0)
        setattr(self, buffer_attribute, mask_prob.copy())

        threshold = float(np.clip(self.config.segmentation_threshold, 0.0, 1.0))
        mask = mask_prob > threshold
        if mask.any() and kernel_size >= 3:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask_uint8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = mask_uint8.astype(bool)

        if not np.any(mask):
            setattr(self, buffer_attribute, None)
            return frame

        color = np.array(self.config.mask_color, dtype=np.float32)
        alpha = float(np.clip(self.config.mask_alpha, 0.0, 1.0))

        if alpha >= 0.999:
            frame[mask] = color.astype(np.uint8)
            return frame

        existing = frame[mask].astype(np.float32)
        blended = alpha * color + (1.0 - alpha) * existing
        frame[mask] = blended.clip(0, 255).astype(np.uint8)
        return frame

    def _generate_face_polygons(
        self,
        faces: Iterable[Any],
        width: int,
        height: int,
    ) -> Iterable[np.ndarray]:
        xmax = max(width - 1, 0)
        ymax = max(height - 1, 0)
        for face_landmarks in faces:
            points: list[tuple[int, int]] = []
            for index in _FACE_OVAL_LANDMARKS:
                try:
                    landmark = face_landmarks.landmark[index]
                except IndexError:
                    continue
                x = int(np.clip(landmark.x * width, 0, xmax))
                y = int(np.clip(landmark.y * height, 0, ymax))
                points.append((x, y))
            if len(points) >= 3:
                yield np.array(points, dtype=np.int32)


__all__ = ["FaceDetectionWindow"]
