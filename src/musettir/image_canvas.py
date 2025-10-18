from __future__ import annotations

from collections import deque
from enum import Enum, auto
from pathlib import Path
from typing import Deque, Iterable, Optional

from PyQt5 import QtCore, QtGui, QtWidgets


class DrawingTool(Enum):
    BRUSH = auto()
    RECTANGLE = auto()


class ImageCanvas(QtWidgets.QWidget):
    """Interactive canvas that supports freehand and rectangular annotations on top of a base image."""

    updated = QtCore.pyqtSignal()
    dropped = QtCore.pyqtSignal(list)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StaticContents)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self._base_pixmap: Optional[QtGui.QPixmap] = None
        self._scaled_pixmap: Optional[QtGui.QPixmap] = None
        self._overlay: Optional[QtGui.QImage] = None
        self._overlay_cache: Optional[QtGui.QPixmap] = None
        self._draw_rect = QtCore.QRectF()
        self._scale_factor = 1.0

        self._tool = DrawingTool.BRUSH
        self._pen_color = QtGui.QColor(255, 0, 0)
        self._pen_width = 28
        self._is_drawing = False
        self._last_point = QtCore.QPointF()
        self._rect_start = QtCore.QPointF()
        self._rect_current = QtCore.QPointF()
        self._max_history = 20
        self._history: Deque[QtGui.QImage] = deque(maxlen=self._max_history)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def has_image(self) -> bool:
        return self._base_pixmap is not None

    def clear(self) -> None:
        self._base_pixmap = None
        self._scaled_pixmap = None
        self._overlay = None
        self._overlay_cache = None
        self._history.clear()
        self._draw_rect = QtCore.QRectF()
        self._scale_factor = 1.0
        self.update()
        self.updated.emit()

    def clear_overlay(self) -> None:
        if self._overlay is None:
            return
        self._overlay.fill(QtCore.Qt.GlobalColor.transparent)
        self._overlay_cache = None
        self._history.clear()
        self.update()
        self.updated.emit()

    def undo(self) -> None:
        if not self._history:
            return
        previous = self._history.pop()
        if self._overlay is not None:
            self._overlay = previous.copy()
            self._overlay_cache = None
            self.update()
            self.updated.emit()

    def set_tool(self, tool: DrawingTool) -> None:
        if tool == self._tool:
            return
        self._tool = tool
        self.setCursor(
            QtCore.Qt.CursorShape.CrossCursor if self._tool == DrawingTool.RECTANGLE else QtCore.Qt.CursorShape.CrossCursor
        )

    def tool(self) -> DrawingTool:
        return self._tool

    def set_pen_color(self, color: QtGui.QColor) -> None:
        if not color.isValid():
            return
        self._pen_color = QtGui.QColor(color)
        self.update()

    def pen_color(self) -> QtGui.QColor:
        return QtGui.QColor(self._pen_color)

    def set_pen_width(self, width: int) -> None:
        width = max(1, min(int(width), 512))
        if width == self._pen_width:
            return
        self._pen_width = width
        self.update()

    def pen_width(self) -> int:
        return int(self._pen_width)

    def set_base_pixmap(self, pixmap: QtGui.QPixmap, *, reset_overlay: bool = False) -> None:
        if pixmap.isNull():
            self.clear()
            return

        same_size = self._base_pixmap is not None and self._base_pixmap.size() == pixmap.size()
        self._base_pixmap = QtGui.QPixmap(pixmap)
        self._scaled_pixmap = None
        if reset_overlay or not same_size:
            self._overlay = QtGui.QImage(
                self._base_pixmap.size(),
                QtGui.QImage.Format.Format_ARGB32_Premultiplied,
            )
            self._overlay.fill(QtCore.Qt.GlobalColor.transparent)
            self._history.clear()
        elif self._overlay is None:
            self._overlay = QtGui.QImage(
                self._base_pixmap.size(),
                QtGui.QImage.Format.Format_ARGB32_Premultiplied,
            )
            self._overlay.fill(QtCore.Qt.GlobalColor.transparent)
        self._overlay_cache = None
        self._update_geometry_cache()
        self.update()
        self.updated.emit()

    def composited_image(self) -> Optional[QtGui.QImage]:
        if self._base_pixmap is None:
            return None
        base = self._base_pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_ARGB32)
        painter = QtGui.QPainter(base)
        if self._overlay is not None:
            painter.drawImage(0, 0, self._overlay)
        painter.end()
        return base

    def sizeHint(self) -> QtCore.QSize:
        if self._base_pixmap is not None:
            return self._base_pixmap.size()
        return QtCore.QSize(960, 600)

    # ------------------------------------------------------------------
    # QWidget overrides
    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(16, 22, 30))

        if self._base_pixmap is None:
            painter.setPen(QtGui.QPen(QtGui.QColor(120, 138, 156)))
            painter.setFont(QtGui.QFont("Montserrat", 16))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "Paste or open an image to begin")
            return

        self._update_geometry_cache()
        target_rect = self._draw_rect.toAlignedRect()
        if target_rect.width() > 0 and target_rect.height() > 0:
            if self._scaled_pixmap is not None:
                painter.drawPixmap(target_rect, self._scaled_pixmap)

        if self._overlay is not None and target_rect.width() > 0 and target_rect.height() > 0:
            if self._overlay_cache is None:
                self._overlay_cache = QtGui.QPixmap.fromImage(self._overlay)
            overlay_scaled = self._overlay_cache.scaled(
                target_rect.size(),
                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(target_rect, overlay_scaled)

        if self._is_drawing and self._tool == DrawingTool.RECTANGLE:
            preview_pen = QtGui.QPen(QtGui.QColor(self._pen_color))
            preview_pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            preview_pen.setWidthF(2)
            painter.setPen(preview_pen)
            mapped_start = self._map_from_image(self._rect_start)
            mapped_current = self._map_from_image(self._rect_current)
            rect = QtCore.QRectF(mapped_start, mapped_current).normalized()
            painter.drawRect(rect)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._scaled_pixmap = None
        self._overlay_cache = None
        self._update_geometry_cache()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.MouseButton.LeftButton or self._base_pixmap is None:
            super().mousePressEvent(event)
            return
        image_point = self._map_to_image(self._event_position(event))
        if image_point is None:
            return

        self._push_history()
        self._is_drawing = True
        if self._tool == DrawingTool.BRUSH:
            self._last_point = image_point
            self._draw_line(self._last_point, image_point)
        else:
            self._rect_start = image_point
            self._rect_current = image_point
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if not self._is_drawing or self._base_pixmap is None:
            super().mouseMoveEvent(event)
            return
        image_point = self._map_to_image(self._event_position(event))
        if image_point is None:
            return
        if self._tool == DrawingTool.BRUSH:
            self._draw_line(self._last_point, image_point)
            self._last_point = image_point
            self.update()
        else:
            self._rect_current = image_point
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if not self._is_drawing:
            return

        self._is_drawing = False
        image_point = self._map_to_image(self._event_position(event))
        if image_point is None:
            image_point = self._rect_current
        if self._tool == DrawingTool.BRUSH:
            self._draw_line(self._last_point, image_point)
        else:
            self._rect_current = image_point
            self._fill_rectangle(self._rect_start, self._rect_current)
        self.update()
        self.updated.emit()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.matches(QtGui.QKeySequence.StandardKey.Undo):
            self.undo()
            return
        if event.matches(QtGui.QKeySequence.StandardKey.Delete):
            self.clear_overlay()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if self._mime_contains_image(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        if not self._mime_contains_image(event.mimeData()):
            super().dropEvent(event)
            return
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        normalized = [str(Path(path)) for path in paths if path]
        if normalized:
            self.dropped.emit(normalized)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _push_history(self) -> None:
        if self._overlay is None:
            return
        snapshot = self._overlay.copy()
        self._history.append(snapshot)

    def _draw_line(self, start: QtCore.QPointF, end: QtCore.QPointF) -> None:
        if self._overlay is None:
            return
        painter = QtGui.QPainter(self._overlay)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        pen = QtGui.QPen(self._pen_color, float(self._pen_width), QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(QtCore.QLineF(start, end))
        painter.end()
        self._overlay_cache = None

    def _fill_rectangle(self, start: QtCore.QPointF, end: QtCore.QPointF) -> None:
        if self._overlay is None:
            return
        rect = QtCore.QRectF(start, end).normalized()
        painter = QtGui.QPainter(self._overlay)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        brush = QtGui.QBrush(self._pen_color)
        painter.fillRect(rect, brush)
        pen = QtGui.QPen(self._pen_color, max(1.0, float(self._pen_width) / 2.0))
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.end()
        self._overlay_cache = None

    def _map_to_image(self, position: QtCore.QPointF) -> Optional[QtCore.QPointF]:
        if self._base_pixmap is None or self._draw_rect.width() <= 0 or self._draw_rect.height() <= 0:
            return None
        if not self._draw_rect.contains(position):
            return None
        x = (position.x() - self._draw_rect.left()) / max(self._scale_factor, 1e-6)
        y = (position.y() - self._draw_rect.top()) / max(self._scale_factor, 1e-6)
        x = float(max(0.0, min(x, self._base_pixmap.width() - 1)))
        y = float(max(0.0, min(y, self._base_pixmap.height() - 1)))
        return QtCore.QPointF(x, y)

    def _map_from_image(self, point: QtCore.QPointF) -> QtCore.QPointF:
        if self._base_pixmap is None:
            return QtCore.QPointF()
        x = self._draw_rect.left() + point.x() * self._scale_factor
        y = self._draw_rect.top() + point.y() * self._scale_factor
        return QtCore.QPointF(x, y)

    @staticmethod
    def _event_position(event: QtGui.QMouseEvent) -> QtCore.QPointF:
        if hasattr(event, "position"):
            pos = event.position()
            return QtCore.QPointF(pos.x(), pos.y())
        local = event.localPos()
        return QtCore.QPointF(local.x(), local.y())

    def _update_geometry_cache(self) -> None:
        if self._base_pixmap is None:
            self._draw_rect = QtCore.QRectF()
            self._scale_factor = 1.0
            self._scaled_pixmap = None
            return

        available = self.rect()
        if available.width() <= 1 or available.height() <= 1:
            self._draw_rect = QtCore.QRectF()
            self._scale_factor = 1.0
            self._scaled_pixmap = None
            return

        scaled = self._base_pixmap.scaled(
            available.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._scaled_pixmap = scaled
        x = available.x() + (available.width() - scaled.width()) / 2
        y = available.y() + (available.height() - scaled.height()) / 2
        self._draw_rect = QtCore.QRectF(x, y, scaled.width(), scaled.height())
        if self._base_pixmap.width() <= 0:
            self._scale_factor = 1.0
        else:
            self._scale_factor = scaled.width() / self._base_pixmap.width()
        self._overlay_cache = None

    @staticmethod
    def _mime_contains_image(mime: Optional[QtCore.QMimeData]) -> bool:
        if mime is None:
            return False
        if mime.hasImage():
            return True
        if not mime.hasUrls():
            return False
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            if Path(url.toLocalFile()).suffix.lower() in ImageCanvas._image_extensions():
                return True
        return False

    @staticmethod
    def _image_extensions() -> Iterable[str]:
        return {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


__all__ = ["DrawingTool", "ImageCanvas"]
