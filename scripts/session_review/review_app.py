#!/usr/bin/env python3
"""FIVRS session review: open a recorded UR5 drilling session and go through it.

One timeline drives everything: the recorded video (mono or stereo), any of the moment-to-moment
curves, and a 3D reconstruction of the bone as it was at that moment, with the drill and its path.

    python3 scripts/session_review/review_app.py [session folder] [--base <recording folder>]

Keys: Space play / pause, Left / Right one frame, Shift + Left / Right one second,
I / O mark the start / end of a stretch, Home / End jump to the ends.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import session_data as data  # noqa: E402

STYLE = """
QWidget { background-color: #0B1C2C; color: #E6EEF5; font-size: 14px; }
QGroupBox { background-color: #12283A; border: 1px solid #1E3A4F; border-radius: 6px; margin-top: 12px;
            padding-top: 8px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 6px; color: #9BB0C3; }
QLabel, QCheckBox { background-color: transparent; }
QTreeWidget, QListWidget { background-color: #0F2433; border: 1px solid #1E3A4F; }
QTreeWidget::item:selected, QListWidget::item:selected { background-color: #1F6F8B; }
QHeaderView::section { background-color: #12283A; color: #9BB0C3; border: none; padding: 4px; }
QComboBox { background-color: #0F2433; border: 1px solid #1E3A4F; border-radius: 4px; padding: 4px 8px; }
QComboBox QAbstractItemView { background-color: #0F2433; selection-background-color: #2F8FBF; }
QPushButton { background-color: #1E3A4F; border: 1px solid #2F8FBF; border-radius: 4px; padding: 6px 12px; }
QPushButton:hover { background-color: #2A4F68; }
QPushButton:disabled { color: #5C7183; border-color: #1E3A4F; }
QPushButton#play { background-color: #1F6F8B; font-weight: bold; min-width: 76px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #2F5A75; border-radius: 3px; background: #0F2433; }
QCheckBox::indicator:checked { background-color: #2F8FBF; border-color: #2F8FBF; }
QSplitter::handle { background-color: #1E3A4F; }
QSlider::groove:horizontal { height: 4px; background: #1E3A4F; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 6px; background: #2F8FBF; }
"""
CURVE_COLORS = ("#6fc3e8", "#f0b35e", "#8fd694", "#e58fb1", "#b9a3ec", "#e8dd7a", "#7fd8c8", "#f08f7f")
DEFAULT_CURVES = ("speed", "force", "voxels", "burr")
LAYOUTS = (("All", (True, True, True)), ("Video", (True, False, False)), ("Video + curves", (True, False, True)),
           ("Video + 3D", (True, True, False)), ("3D + curves", (False, True, True)), ("3D", (False, True, False)),
           ("Curves", (False, False, True)))
CURVE_WINDOWS = (("Whole session", 0.0), ("60 s", 60.0), ("30 s", 30.0), ("10 s", 10.0), ("3 s", 3.0))


def clock(seconds):
    seconds = max(seconds, 0.0)
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def number(value):
    if value is None or not np.isfinite(value):
        return "-"
    magnitude = abs(value)
    if float(value).is_integer():
        return f"{value:,.0f}"
    if magnitude >= 1000:
        return f"{value:,.0f}"
    return f"{value:.1f}" if magnitude >= 10 else f"{value:.2f}"


class VideoView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(200, 140)
        self.images = []
        self.message = "No session loaded"

    def show_images(self, images, message=""):
        self.images = images
        self.message = message
        self.update()

    def paintEvent(self, _):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#050B12"))
        if not self.images:
            painter.setPen(QtGui.QColor("#9BB0C3"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        cell = self.width() / len(self.images)
        for k, image in enumerate(self.images):
            height, width = image.shape[:2]
            scale = min(cell / width, self.height() / height)
            w, h = width * scale, height * scale
            target = QtCore.QRectF(k * cell + (cell - w) / 2, (self.height() - h) / 2, w, h)
            qimage = QtGui.QImage(image.data, width, height, 3 * width, QtGui.QImage.Format_BGR888)
            painter.drawImage(target, qimage)


class CurvePanel(QtWidgets.QWidget):
    seek = pyqtSignal(float)
    GUTTER = 214

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.session = None
        self.keys = []
        self.span = (0.0, 1.0)
        self.cursor_t = 0.0
        self._cache = None
        self._cache_key = None
        self.setMouseTracking(False)

    def set_session(self, session):
        self.session = session
        self._cache_key = None
        self.update()

    def set_keys(self, keys):
        self.keys = list(keys)
        self._cache_key = None
        self.update()

    def set_view(self, start, stop, cursor):
        self.span, self.cursor_t = (start, max(stop, start + 1e-3)), cursor
        self.update()

    def _x(self, t):
        start, stop = self.span
        return self.GUTTER + (t - start) / (stop - start) * max(self.width() - self.GUTTER - 8, 1)

    def _draw_curve(self, painter, channel, lane, color):
        start, stop = self.span
        t, y = channel.t, channel.y
        first = max(int(np.searchsorted(t, start)) - 1, 0)
        last = min(int(np.searchsorted(t, stop)) + 1, len(t))
        if last - first < 1:
            return
        t, y = t[first:last], y[first:last]
        finite = y[np.isfinite(y)]
        if not len(finite):
            return
        low, high = float(finite.min()), float(finite.max())
        if high - low < 1e-9:
            low, high = low - 0.5, high + 0.5
        top, bottom = lane.top() + 5, lane.bottom() - 5
        to_y = lambda v: bottom - (v - low) / (high - low) * (bottom - top)
        painter.setPen(QtGui.QPen(QtGui.QColor(color), 1.2))
        pixels = max(int(lane.width()), 2)
        if len(t) > pixels * 2 and not channel.step:
            edges = np.linspace(start, stop, pixels + 1)[:-1]
            cuts = np.unique(np.clip(np.searchsorted(t, edges), 0, len(t) - 1))
            lows, highs = np.minimum.reduceat(y, cuts), np.maximum.reduceat(y, cuts)
            xs = self.GUTTER + (t[cuts] - start) / (stop - start) * lane.width()
            lines = [QtCore.QLineF(x, to_y(a), x, min(to_y(b), to_y(a) - 0.6)) for x, a, b in zip(xs, lows, highs)]
            painter.drawLines(lines)
        else:
            if len(t) > pixels * 2:  # a step curve: keep only its changes
                keep = np.concatenate([[True], y[1:] != y[:-1]])
                t, y = t[keep], y[keep]
            points = []
            for k in range(len(t)):
                x = self._x(float(t[k]))
                if channel.step and k:
                    points.append(QtCore.QPointF(x, to_y(float(y[k - 1]))))
                points.append(QtCore.QPointF(x, to_y(float(y[k]))))
            if channel.step:
                points.append(QtCore.QPointF(self._x(stop), to_y(float(y[-1]))))
            painter.drawPolyline(QtGui.QPolygonF(points))
        painter.setPen(QtGui.QColor("#6F879A"))
        small = painter.font()
        small.setPointSize(8)
        painter.setFont(small)
        painter.drawText(QtCore.QRectF(lane.left() + 3, lane.top() + 1, 90, 14), Qt.AlignLeft, number(high))
        painter.drawText(QtCore.QRectF(lane.left() + 3, lane.bottom() - 15, 90, 14), Qt.AlignLeft, number(low))

    def _render(self):
        ratio = self.devicePixelRatioF()
        pixmap = QtGui.QPixmap(int(self.width() * ratio), int(self.height() * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QtGui.QColor("#0F2433"))
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        keys = [k for k in self.keys if self.session is not None and k in self.session.channels]
        if keys:
            lane_h = self.height() / len(keys)
            start, stop = self.span
            for span_start, span_stop in self.session.stroke_spans():
                if span_stop >= start and span_start <= stop:
                    a, b = self._x(max(span_start, start)), self._x(min(span_stop, stop))
                    painter.fillRect(QtCore.QRectF(a, 0, max(b - a, 1), self.height()), QtGui.QColor(47, 143, 191, 26))
            for k, key in enumerate(keys):
                lane = QtCore.QRectF(self.GUTTER, k * lane_h, self.width() - self.GUTTER - 8, lane_h)
                painter.setPen(QtGui.QColor("#1E3A4F"))
                painter.drawLine(QtCore.QLineF(0, lane.bottom(), self.width(), lane.bottom()))
                self._draw_curve(painter, self.session.channels[key], lane, CURVE_COLORS[k % len(CURVE_COLORS)])
        painter.end()
        return pixmap

    def paintEvent(self, _):
        painter = QtGui.QPainter(self)
        key = (tuple(self.keys), self.span, self.width(), self.height(), id(self.session))
        if key != self._cache_key:
            self._cache, self._cache_key = self._render(), key
        painter.drawPixmap(0, 0, self._cache)
        keys = [k for k in self.keys if self.session is not None and k in self.session.channels]
        if not keys:
            painter.setPen(QtGui.QColor("#9BB0C3"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Choose curves on the left")
            return
        lane_h = self.height() / len(keys)
        for k, name in enumerate(keys):
            channel = self.session.channels[name]
            color = QtGui.QColor(CURVE_COLORS[k % len(CURVE_COLORS)])
            value = channel.value_at(self.cursor_t)
            text = number(value)
            if name in ("on", "device_clutch", "cam_clutch", "paused") and value is not None:
                text = "on" if value else "off"
            painter.setPen(color)
            font = painter.font()
            font.setPointSize(9)
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(QtCore.QRectF(8, k * lane_h + 3, self.GUTTER - 12, 17), Qt.AlignLeft, channel.label)
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QtGui.QColor("#E6EEF5"))
            unit = f" {channel.unit}" if channel.unit and text not in ("on", "off") else ""
            painter.drawText(QtCore.QRectF(8, k * lane_h + 20, self.GUTTER - 12, 22), Qt.AlignLeft, text + unit)
        if self.span[0] <= self.cursor_t <= self.span[1]:
            x = self._x(self.cursor_t)
            painter.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF"), 1))
            painter.drawLine(QtCore.QLineF(x, 0, x, self.height()))

    def _mouse(self, event):
        if self.session is None:
            return
        fraction = (event.x() - self.GUTTER) / max(self.width() - self.GUTTER - 8, 1)
        self.seek.emit(self.span[0] + np.clip(fraction, 0.0, 1.0) * (self.span[1] - self.span[0]))

    def mousePressEvent(self, event):
        self._mouse(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._mouse(event)


class Timeline(QtWidgets.QWidget):
    seek = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(78)
        self.session = None
        self.t = 0.0
        self.mark = (None, None)

    def _x(self, t):
        return 10 + t / max(self.session.duration, 1e-3) * max(self.width() - 20, 1)

    def paintEvent(self, _):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#0F2433"))
        session = self.session
        if session is None:
            return
        small = painter.font()
        small.setPointSize(8)
        painter.setFont(small)
        burr_rect_y, stroke_y, tick_y = 5, 29, 49
        for start, stop, size, kind in session.burr_spans():
            a, b = self._x(start), self._x(stop)
            rect = QtCore.QRectF(a, burr_rect_y, max(b - a, 1), 20)
            painter.fillRect(rect, QtGui.QColor(data.BURR_COLORS.get(size, "#9BB0C3")))
            if b - a > 54:
                painter.setPen(QtGui.QColor("#0B1C2C"))
                painter.drawText(rect, Qt.AlignCenter, f"{size} mm {kind}")
        painter.fillRect(QtCore.QRectF(10, stroke_y, self.width() - 20, 14), QtGui.QColor("#12283A"))
        for start, stop in session.stroke_spans():
            a, b = self._x(start), self._x(stop)
            painter.fillRect(QtCore.QRectF(a, stroke_y, max(b - a, 1.5), 14), QtGui.QColor("#2F8FBF"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#F07F7F"), 2))
        for reset in session.voxels.reset_times:
            if 0 <= reset <= session.duration:
                painter.drawLine(QtCore.QLineF(self._x(reset), burr_rect_y, self._x(reset), stroke_y + 14))
        painter.setPen(QtGui.QColor("#6F879A"))
        step = next((s for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600) if session.duration / s <= 12), 1200)
        for tick in np.arange(0, session.duration + 1e-6, step):
            x = self._x(float(tick))
            painter.drawLine(QtCore.QLineF(x, tick_y, x, tick_y + 4))
            painter.drawText(QtCore.QRectF(x - 30, tick_y + 5, 60, 14), Qt.AlignCenter, f"{int(tick // 60)}:{int(tick % 60):02d}")
        start, stop = self.mark
        if start is not None or stop is not None:
            a = self._x(start if start is not None else 0.0)
            b = self._x(stop if stop is not None else session.duration)
            painter.fillRect(QtCore.QRectF(a, 0, b - a, self.height()), QtGui.QColor(240, 179, 94, 42))
            painter.setPen(QtGui.QPen(QtGui.QColor("#F0B35E"), 1.5))
            painter.drawLine(QtCore.QLineF(a, 0, a, self.height()))
            painter.drawLine(QtCore.QLineF(b, 0, b, self.height()))
        x = self._x(self.t)
        painter.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF"), 2))
        painter.drawLine(QtCore.QLineF(x, 0, x, self.height()))

    def _mouse(self, event):
        if self.session is not None:
            fraction = (event.x() - 10) / max(self.width() - 20, 1)
            self.seek.emit(float(np.clip(fraction, 0.0, 1.0) * self.session.duration))

    def mousePressEvent(self, event):
        self._mouse(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._mouse(event)


class Scene3D(QtWidgets.QWidget):
    """The bone at the current moment, rebuilt from the recorded voxels."""

    def __init__(self):
        super().__init__()
        self.session = None
        self.available = False
        self.cost_s = 0.03
        self._voxel_position = None
        self._t = 0.0
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QtWidgets.QHBoxLayout()
        self.check_bone = QtWidgets.QCheckBox("Bone")
        self.check_ear = QtWidgets.QCheckBox("Inner ear")
        self.check_removed = QtWidgets.QCheckBox("Removed")
        self.check_path = QtWidgets.QCheckBox("Path")
        self.check_follow = QtWidgets.QCheckBox("Microscope view")
        for widget in (self.check_bone, self.check_ear, self.check_removed, self.check_path, self.check_follow):
            widget.setChecked(True)
            widget.toggled.connect(self._options_changed)
            bar.addWidget(widget)
        bar.addWidget(QtWidgets.QLabel("Bone opacity"))
        self.opacity = QtWidgets.QSlider(Qt.Horizontal)
        self.opacity.setRange(10, 100)
        self.opacity.setValue(100)
        self.opacity.setMaximumWidth(110)
        self.opacity.valueChanged.connect(self._options_changed)
        bar.addWidget(self.opacity)
        self.button_fit = QtWidgets.QPushButton("Fit")
        self.button_fit.clicked.connect(self.fit)
        bar.addWidget(self.button_fit)
        bar.addStretch(1)
        layout.addLayout(bar)
        if os.environ.get("FIVRS_REVIEW_NO_3D"):
            layout.addWidget(QtWidgets.QLabel("3D view off"), 1)
            return
        try:
            import vtk
            from vtk.util import numpy_support
            from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        except ImportError:
            note = QtWidgets.QLabel("The 3D view needs VTK:  sudo apt install python3-vtk9")
            note.setAlignment(Qt.AlignCenter)
            layout.addWidget(note, 1)
            return
        self.vtk, self.np_vtk = vtk, numpy_support
        self.widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.widget, 1)
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.03, 0.06, 0.09)
        self.widget.GetRenderWindow().AddRenderer(self.renderer)
        # The drill path is drawn in a layer on top, so it stays visible inside the cavity.
        self.overlay = vtk.vtkRenderer()
        self.overlay.SetLayer(1)
        self.overlay.InteractiveOff()
        self.overlay.SetActiveCamera(self.renderer.GetActiveCamera())
        self.widget.GetRenderWindow().SetNumberOfLayers(2)
        self.widget.GetRenderWindow().AddRenderer(self.overlay)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.widget.SetInteractorStyle(style)
        # Dragging the view by hand takes it off the recorded microscope.
        style.AddObserver("StartInteractionEvent", lambda *_: self.check_follow.setChecked(False))
        light = vtk.vtkLightKit()
        light.AddLightsToRenderer(self.renderer)
        self.surfaces = {}
        self.available = True
        self._build_instruments()

    def showEvent(self, event):
        super().showEvent(event)
        if self.available and not getattr(self, "_started", False):
            self._started = True
            self.widget.Initialize()

    def _build_instruments(self):
        vtk = self.vtk
        self.burr_source = vtk.vtkSphereSource()
        self.burr_source.SetThetaResolution(28)
        self.burr_source.SetPhiResolution(20)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.burr_source.GetOutputPort())
        self.burr_actor = vtk.vtkActor()
        self.burr_actor.SetMapper(mapper)
        self.burr_actor.GetProperty().SetColor(0.95, 0.72, 0.25)
        self.burr_actor.GetProperty().SetSpecular(0.6)
        self.shaft_source = vtk.vtkCylinderSource()
        self.shaft_source.SetResolution(20)
        self.shaft_source.SetHeight(1.2)
        shaft_mapper = vtk.vtkPolyDataMapper()
        shaft_mapper.SetInputConnection(self.shaft_source.GetOutputPort())
        self.shaft_actor = vtk.vtkActor()
        self.shaft_actor.SetMapper(shaft_mapper)
        self.shaft_actor.GetProperty().SetColor(0.62, 0.68, 0.74)
        self.path_data = vtk.vtkPolyData()
        path_mapper = vtk.vtkPolyDataMapper()
        path_mapper.SetInputData(self.path_data)
        self.path_actor = vtk.vtkActor()
        self.path_actor.SetMapper(path_mapper)
        self.path_actor.GetProperty().SetColor(0.3, 0.85, 1.0)
        self.path_actor.GetProperty().SetLineWidth(1.5)
        self.path_actor.GetProperty().LightingOff()
        for actor in (self.burr_actor, self.shaft_actor):
            self.renderer.AddActor(actor)
        self.overlay.AddActor(self.path_actor)

    def _surface(self, name, color, shape, volume, smooth=True):
        vtk = self.vtk
        buffer = np.zeros(int(np.prod(shape)), np.uint8)
        image = vtk.vtkImageData()
        image.SetDimensions(*shape)
        dims = np.asarray(volume["dimensions_sim"], float)
        image.SetSpacing(*(dims / np.asarray(shape, float)))
        image.SetOrigin(*(-dims / 2.0))
        scalars = self.np_vtk.numpy_to_vtk(buffer, deep=0, array_type=vtk.VTK_UNSIGNED_CHAR)
        image.GetPointData().SetScalars(scalars)
        contour = vtk.vtkFlyingEdges3D()
        if smooth:
            blur = vtk.vtkImageGaussianSmooth()
            blur.SetInputData(image)
            blur.SetStandardDeviations(0.8, 0.8, 0.8)
            blur.SetRadiusFactors(1.6, 1.6, 1.6)
            contour.SetInputConnection(blur.GetOutputPort())
        else:
            contour.SetInputData(image)
        contour.SetValue(0, 127.0)
        contour.ComputeNormalsOn()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(contour.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetSpecular(0.12)
        matrix = vtk.vtkMatrix4x4()
        rotation, position = np.asarray(volume["R_world_volume"]), volume["p_world_volume"]
        for r in range(3):
            for c in range(3):
                matrix.SetElement(r, c, float(rotation[r, c]))
            matrix.SetElement(r, 3, float(position[r]))
        actor.SetUserMatrix(matrix)
        self.renderer.AddActor(actor)
        self.surfaces[name] = {"buffer": buffer, "image": image, "actor": actor, "scalars": scalars}

    def set_session(self, session, t):
        self.session = session
        self._voxel_position = None
        if not self.available:
            return
        for surface in self.surfaces.values():
            self.renderer.RemoveActor(surface["actor"])
        self.surfaces = {}
        labels = session.voxels.labels_at(t) if session.voxels.epochs else None
        if labels is not None:
            volume = session.meta["volume"]
            self._surface("bone", (0.93, 0.89, 0.76), labels.shape, volume)
            self._surface("ear", (0.86, 0.26, 0.20), labels.shape, volume)
            self._surface("removed", (0.25, 0.65, 0.95), labels.shape, volume)
        self._options_changed()
        self.update_time(t, force=True)
        if not self.check_follow.isChecked() or session.camera_at(t) is None:
            self.fit()

    def fit(self):
        if self.available:
            self.renderer.ResetCamera()
            self.renderer.ResetCameraClippingRange()
            self.widget.GetRenderWindow().Render()

    def _options_changed(self, *_):
        if not self.available:
            return
        shown = {"bone": self.check_bone, "ear": self.check_ear, "removed": self.check_removed}
        for name, surface in self.surfaces.items():
            surface["actor"].SetVisibility(shown[name].isChecked())
        if "bone" in self.surfaces:
            self.surfaces["bone"]["actor"].GetProperty().SetOpacity(self.opacity.value() / 100.0)
        if "removed" in self.surfaces:
            self.surfaces["removed"]["actor"].GetProperty().SetOpacity(0.4)
        self.path_actor.SetVisibility(self.check_path.isChecked())
        if self.session is not None:
            self.update_time(self._t, force=True)

    def update_time(self, t, force=False):
        if not self.available or self.session is None:
            return
        began = time.perf_counter()
        self._t = t
        session = self.session
        position = session.voxels.position_at(t) if session.voxels.epochs else None
        if self.surfaces and (force or position != self._voxel_position):
            self._voxel_position = position
            labels = session.voxels.labels_at(t)
            removed = session.voxels.removed_at(t)
            for name, mask in (("bone", (labels == 1) | (labels == 3)), ("ear", labels == 2), ("removed", removed)):
                surface = self.surfaces[name]
                surface["buffer"][:] = (mask.ravel(order="F") * 255).astype(np.uint8)
                surface["scalars"].Modified()
                surface["image"].Modified()
        self._place_drill(t)
        if self.check_follow.isChecked():
            self._follow_microscope(t)
        self.renderer.ResetCameraClippingRange()
        self.widget.GetRenderWindow().Render()
        self.cost_s = 0.8 * self.cost_s + 0.2 * (time.perf_counter() - began)

    def _place_drill(self, t):
        vtk = self.vtk
        session = self.session
        pose = session.drill_at(t)
        _, _, diameter = session.burr_at(t)
        self.burr_source.SetRadius(diameter * 0.5)
        rotation = data.dataset.quat_to_matrix(pose[3:7])
        position = pose[:3]
        for actor, offset in ((self.burr_actor, 0.0), (self.shaft_actor, 0.6)):
            matrix = vtk.vtkMatrix4x4()
            # The drill's shaft runs along its local +x from the burr. VTK's cylinder is along y.
            axes = rotation if actor is self.burr_actor else rotation @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)
            origin = position + rotation[:, 0] * offset
            for r in range(3):
                for c in range(3):
                    matrix.SetElement(r, c, float(axes[r, c]))
                matrix.SetElement(r, 3, float(origin[r]))
            actor.SetUserMatrix(matrix)
        self.shaft_source.SetRadius(max(diameter * 0.22, 0.006))
        if self.check_path.isChecked():
            points = session.path_until(t)
            polydata_points = vtk.vtkPoints()
            polydata_points.SetData(self.np_vtk.numpy_to_vtk(points.astype(np.float32), deep=1))
            lines = vtk.vtkCellArray()
            if len(points) > 1:
                lines.InsertNextCell(len(points))
                for k in range(len(points)):
                    lines.InsertCellPoint(k)
            self.path_data.SetPoints(polydata_points)
            self.path_data.SetLines(lines)
            self.path_data.Modified()

    def _follow_microscope(self, t):
        camera = self.session.camera_at(t)
        if camera is None:
            return
        position, look, up, fov = camera
        target = np.asarray(self.session.meta["volume"]["p_world_volume"], float)
        distance = max(float(np.dot(target - position, look)), 0.1)
        view = self.renderer.GetActiveCamera()
        view.SetPosition(*position)
        view.SetFocalPoint(*(position + look * distance))
        view.SetViewUp(*up)
        view.SetViewAngle(float(np.degrees(fov)))

    def shutdown(self):
        if self.available:
            self.widget.Finalize()


class ReviewWindow(QtWidgets.QWidget):
    def __init__(self, base, embedded=False):
        super().__init__()
        self.embedded = embedded  # a tab in the launcher: keys must not reach its other tabs
        self._shut_down = False
        self.base = pathlib.Path(base)
        self.session = None
        self.t = 0.0
        self.playing = False
        self.mark = [None, None]
        self._last_tick = 0.0
        self._last_3d = 0.0
        self._shown_frame = None
        self.setWindowTitle("FIVRS Session Review")
        self.setStyleSheet(STYLE)
        self._build()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.refresh_sessions()

    # ---------------------------------------------------------------- layout
    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        outer = QtWidgets.QSplitter(Qt.Horizontal)
        root.addWidget(outer)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 4, 0)
        row = QtWidgets.QHBoxLayout()
        self.button_open = QtWidgets.QPushButton("Open folder...")
        self.button_open.clicked.connect(self.open_folder)
        self.button_refresh = QtWidgets.QPushButton("Refresh")
        self.button_refresh.clicked.connect(self.refresh_sessions)
        row.addWidget(self.button_open)
        row.addWidget(self.button_refresh)
        side_layout.addLayout(row)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Session", "Length"])
        self.tree.setColumnWidth(0, 205)
        self.tree.itemActivated.connect(self._tree_open)
        self.tree.itemClicked.connect(self._tree_open)
        side_layout.addWidget(self.tree, 3)
        box = QtWidgets.QGroupBox("Session")
        box_layout = QtWidgets.QVBoxLayout(box)
        self.info = QtWidgets.QLabel("Pick a session.")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        box_layout.addWidget(self.info)
        side_layout.addWidget(box)
        curves = QtWidgets.QGroupBox("Curves")
        curves_layout = QtWidgets.QVBoxLayout(curves)
        self.curve_list = QtWidgets.QListWidget()
        self.curve_list.itemChanged.connect(self._curves_changed)
        curves_layout.addWidget(self.curve_list)
        side_layout.addWidget(curves, 3)
        outer.addWidget(side)

        main = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main)
        main_layout.setContentsMargins(4, 0, 0, 0)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Show:"))
        self.combo_layout = QtWidgets.QComboBox()
        for name, _ in LAYOUTS:
            self.combo_layout.addItem(name)
        self.combo_layout.currentIndexChanged.connect(self._layout_changed)
        top.addWidget(self.combo_layout)
        self.label_eye = QtWidgets.QLabel("Eye:")
        self.combo_eye = QtWidgets.QComboBox()
        self.combo_eye.currentIndexChanged.connect(lambda *_: self._show_video(force=True))
        top.addWidget(self.label_eye)
        top.addWidget(self.combo_eye)
        top.addWidget(QtWidgets.QLabel("Curve window:"))
        self.combo_window = QtWidgets.QComboBox()
        for name, seconds in CURVE_WINDOWS:
            self.combo_window.addItem(name, seconds)
        self.combo_window.currentIndexChanged.connect(lambda *_: self._update_curves())
        top.addWidget(self.combo_window)
        top.addStretch(1)
        self.button_export_video = QtWidgets.QPushButton("Export video...")
        self.button_export_csv = QtWidgets.QPushButton("Export curves (CSV)...")
        self.button_export_video.clicked.connect(self.export_video)
        self.button_export_csv.clicked.connect(self.export_csv)
        top.addWidget(self.button_export_video)
        top.addWidget(self.button_export_csv)
        main_layout.addLayout(top)

        self.vertical = QtWidgets.QSplitter(Qt.Vertical)
        self.upper = QtWidgets.QSplitter(Qt.Horizontal)
        self.video_view = VideoView()
        self.scene = Scene3D()
        self.upper.addWidget(self.video_view)
        self.upper.addWidget(self.scene)
        self.curve_panel = CurvePanel()
        self.curve_panel.seek.connect(self.seek)
        self.vertical.addWidget(self.upper)
        self.vertical.addWidget(self.curve_panel)
        self.vertical.setStretchFactor(0, 3)
        self.vertical.setStretchFactor(1, 2)
        main_layout.addWidget(self.vertical, 1)

        self.timeline = Timeline()
        self.timeline.seek.connect(self.seek)
        main_layout.addWidget(self.timeline)
        transport = QtWidgets.QHBoxLayout()
        self.button_play = QtWidgets.QPushButton("Play")
        self.button_play.setObjectName("play")
        self.button_play.clicked.connect(self.toggle_play)
        transport.addWidget(self.button_play)
        for text, step in (("-1 s", -1.0), ("-1 frame", "back"), ("+1 frame", "next"), ("+1 s", 1.0)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _=False, s=step: self.step(s))
            transport.addWidget(button)
        transport.addWidget(QtWidgets.QLabel("Speed:"))
        self.combo_speed = QtWidgets.QComboBox()
        for speed in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            self.combo_speed.addItem(f"{speed:g}x", speed)
        self.combo_speed.setCurrentIndex(3)
        transport.addWidget(self.combo_speed)
        self.check_loop = QtWidgets.QCheckBox("Loop")
        transport.addWidget(self.check_loop)
        self.label_time = QtWidgets.QLabel("00:00.000 / 00:00.000")
        font = self.label_time.font()
        font.setPointSize(12)
        font.setBold(True)
        self.label_time.setFont(font)
        transport.addWidget(self.label_time)
        self.label_now = QtWidgets.QLabel("")
        transport.addWidget(self.label_now)
        transport.addStretch(1)
        self.button_in = QtWidgets.QPushButton("Mark start (I)")
        self.button_out = QtWidgets.QPushButton("Mark end (O)")
        self.button_clear = QtWidgets.QPushButton("Clear")
        self.button_in.clicked.connect(lambda: self.set_mark(0))
        self.button_out.clicked.connect(lambda: self.set_mark(1))
        self.button_clear.clicked.connect(self.clear_marks)
        for button in (self.button_in, self.button_out, self.button_clear):
            transport.addWidget(button)
        main_layout.addLayout(transport)
        outer.addWidget(main)
        outer.setStretchFactor(0, 0)
        outer.setStretchFactor(1, 1)
        outer.setSizes([330, 1270])
        for key, action in ((Qt.Key_Space, self.toggle_play), (Qt.Key_Left, lambda: self.step("back")),
                            (Qt.Key_Right, lambda: self.step("next")), (Qt.SHIFT + Qt.Key_Left, lambda: self.step(-1.0)),
                            (Qt.SHIFT + Qt.Key_Right, lambda: self.step(1.0)), (Qt.Key_I, lambda: self.set_mark(0)),
                            (Qt.Key_O, lambda: self.set_mark(1)), (Qt.Key_Home, lambda: self.seek(0.0)),
                            (Qt.Key_End, lambda: self.seek(self.session.duration if self.session else 0.0))):
            QtWidgets.QShortcut(QtGui.QKeySequence(key), self, activated=action,
                                context=Qt.WidgetWithChildrenShortcut if self.embedded else Qt.WindowShortcut)
        self._set_enabled(False)

    def _set_enabled(self, enabled):
        for widget in (self.button_play, self.button_in, self.button_out, self.button_clear, self.button_export_csv):
            widget.setEnabled(enabled)
        has_video = enabled and self.session is not None and self.session.video is not None and self.session.video.count > 0
        self.button_export_video.setEnabled(has_video)

    def _layout_changed(self, *_):
        video, scene, curves = LAYOUTS[self.combo_layout.currentIndex()][1]
        self.video_view.setVisible(video)
        self.scene.setVisible(scene)
        self.upper.setVisible(video or scene)
        self.curve_panel.setVisible(curves)
        self._refresh(force=True)

    # ---------------------------------------------------------------- sessions
    def refresh_sessions(self):
        self.tree.clear()
        found = data.discover(self.base)
        groups = {}
        for entry in found:
            kind = entry["kind"]
            top = groups.setdefault(kind, {"item": QtWidgets.QTreeWidgetItem([kind]), "people": {}})
            person = top["people"].setdefault(entry["person"], {"item": QtWidgets.QTreeWidgetItem([entry["person"]]), "specimens": {}})
            if person["item"].parent() is None:
                top["item"].addChild(person["item"])
            specimen = person["specimens"].get(entry["specimen"])
            if specimen is None:
                specimen = person["specimens"][entry["specimen"]] = QtWidgets.QTreeWidgetItem([entry["specimen"]])
                person["item"].addChild(specimen)
            duration = entry["summary"].get("duration_s")
            length = clock(duration)[:-4] if duration is not None else "unfinished"
            leaf = QtWidgets.QTreeWidgetItem([f"Session {entry['number']}", length])
            leaf.setData(0, Qt.UserRole, str(entry["folder"]))
            leaf.setToolTip(0, entry["started"])
            specimen.addChild(leaf)
        for kind in ("Expert", "Main", "Development"):
            if kind in groups:
                self.tree.addTopLevelItem(groups[kind]["item"])
        self.tree.expandAll()
        if not found:
            self.info.setText(f"No recorded sessions yet in {self.base}.")

    def _tree_open(self, item, *_):
        folder = item.data(0, Qt.UserRole)
        if folder and (self.session is None or str(self.session.folder) != folder):
            self.load(folder)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open a session folder", str(self.base))
        if folder:
            self.load(folder)

    def load(self, folder):
        self.pause()
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            session = data.ReviewSession(folder)
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Session review", f"Could not open this session:\n{exc}")
            return
        try:
            if self.session is not None:
                self.session.close()
            self.session = session
            self.t = 0.0
            self.mark = [None, None]
            self._shown_frame = None
            self.timeline.session = session
            self.timeline.mark = (None, None)
            self.curve_panel.set_session(session)
            self._fill_curves()
            self._fill_eyes()
            self._show_info()
            self.scene.set_session(session, 0.0)
            self._set_enabled(True)
            self._refresh(force=True)
            self.setWindowTitle(f"FIVRS Session Review: {session.meta.get('specimen', {}).get('name', '')}, "
                                f"session {session.meta.get('session_number', '')}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _fill_eyes(self):
        video = self.session.video
        self.combo_eye.blockSignals(True)
        self.combo_eye.clear()
        stereo = video is not None and "right" in video.eyes
        if stereo:
            self.combo_eye.addItem("Both", "both")
            self.combo_eye.addItem("Left", "left")
            self.combo_eye.addItem("Right", "right")
        self.combo_eye.blockSignals(False)
        self.combo_eye.setVisible(stereo)
        self.label_eye.setVisible(stereo)

    def _fill_curves(self):
        self.curve_list.blockSignals(True)
        self.curve_list.clear()
        group = None
        for key, channel in self.session.channels.items():
            if channel.group != group:
                group = channel.group
                header = QtWidgets.QListWidgetItem(group)
                header.setFlags(Qt.NoItemFlags)
                header.setForeground(QtGui.QColor("#9BB0C3"))
                self.curve_list.addItem(header)
            item = QtWidgets.QListWidgetItem("  " + channel.label)
            item.setData(Qt.UserRole, key)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if key in DEFAULT_CURVES else Qt.Unchecked)
            self.curve_list.addItem(item)
        self.curve_list.blockSignals(False)
        self._curves_changed()

    def selected_curves(self):
        keys = []
        for k in range(self.curve_list.count()):
            item = self.curve_list.item(k)
            if item.data(Qt.UserRole) and item.checkState() == Qt.Checked:
                keys.append(item.data(Qt.UserRole))
        return keys

    def _curves_changed(self, *_):
        self.curve_panel.set_keys(self.selected_curves())
        self._update_curves()

    def _show_info(self):
        session = self.session
        meta, summary = session.meta, session.summary
        who = meta.get("participant") or {}
        specimen = meta.get("specimen") or {}
        check = (summary.get("replay_check") or {}).get("status", "not run")
        verdict = {"exact": "exact match", "mismatch": "MISMATCH", "not_checked": "not checked"}.get(check, check)
        video = session.video
        video_text = "none"
        if video is not None and video.count:
            video_text = f"{'stereo' if 'right' in video.eyes else 'mono'}, {video.count:,} frames, {video.fps:.0f} fps"
            if video.dropped:
                video_text += f", {video.dropped} dropped"
        burrs = " → ".join(str(b["size_mm"]) for b in summary.get("burr_sequence", [])) or "-"
        lines = [
            f"<b>{who.get('first_name', '')} {who.get('last_name', '')}</b>"
            + (" <span style='color:#F0B35E'>(development)</span>" if meta.get("development") else ""),
            f"{specimen.get('name', '')}, {specimen.get('laterality', '')} ear, session {meta.get('session_number', '')}",
            f"{meta.get('started', '')}",
            f"Length {clock(session.duration)[:-4]}, {summary.get('strokes', 0)} strokes",
            f"Voxels {summary.get('voxels_removed', 0):,} (inner ear {summary.get('inner_ear_voxels_removed', 0):,})",
            f"Burr {burrs} mm",
            f"Video {video_text}",
            f"Replay check: <b>{verdict}</b>",
        ]
        if not summary:
            lines.append("<span style='color:#F07F7F'>Recording did not finish.</span>")
        self.info.setText("<br>".join(lines))

    # ---------------------------------------------------------------- time
    def toggle_play(self):
        if self.session is None:
            return
        if self.playing:
            self.pause()
        else:
            start, stop = self._range()
            if self.t >= stop - 1e-3:
                self.t = start
            self.playing = True
            self._last_tick = time.monotonic()
            self.button_play.setText("Pause")
            self.timer.start()

    def pause(self):
        self.playing = False
        self.timer.stop()
        self.button_play.setText("Play")
        if self.session is not None:
            self._refresh(force=True)

    def _range(self):
        start = self.mark[0] if self.mark[0] is not None else 0.0
        stop = self.mark[1] if self.mark[1] is not None else self.session.duration
        return start, stop

    def _tick(self):
        now = time.monotonic()
        dt, self._last_tick = now - self._last_tick, now
        self.t += min(dt, 0.25) * float(self.combo_speed.currentData())
        start, stop = self._range()
        if self.t >= stop:
            if self.check_loop.isChecked():
                self.t = start
            else:
                self.t = stop
                self.pause()
                return
        self._refresh()

    def seek(self, t):
        if self.session is None:
            return
        self.t = float(np.clip(t, 0.0, self.session.duration))
        self._refresh(force=not self.playing)

    def step(self, amount):
        if self.session is None:
            return
        self.pause()
        video = self.session.video
        if amount in ("back", "next"):
            if video is not None and video.count:
                index = video.index_at(self.t) + (1 if amount == "next" else -1)
                index = int(np.clip(index, 0, video.count - 1))
                self.seek(max(float(video.t[index]), 0.0))
            else:
                self.seek(self.t + (0.01 if amount == "next" else -0.01))
        else:
            self.seek(self.t + amount)

    def set_mark(self, which):
        if self.session is None:
            return
        self.mark[which] = self.t
        if self.mark[0] is not None and self.mark[1] is not None and self.mark[0] > self.mark[1]:
            self.mark = [self.mark[1], self.mark[0]]
        self.timeline.mark = tuple(self.mark)
        self.timeline.update()

    def clear_marks(self):
        self.mark = [None, None]
        self.timeline.mark = (None, None)
        self.timeline.update()

    # ---------------------------------------------------------------- drawing
    def _refresh(self, force=False):
        session = self.session
        if session is None:
            return
        self.timeline.t = self.t
        self.timeline.update()
        self.label_time.setText(f"{clock(self.t)} / {clock(session.duration)}")
        size, kind, _ = session.burr_at(self.t)
        on = session.value_at("on", self.t)
        force_value = session.value_at("force", self.t)
        speed = session.value_at("speed", self.t)
        self.label_now.setText(f"{size} mm {kind}   drill {'on' if on else 'off'}   "
                               f"{number(force_value)} N   {number(speed)} mm/s")
        self._show_video(force=force)
        self._update_curves()
        if self.scene.isVisible():
            now = time.monotonic()
            # The 3D view takes what time is left: it never holds the video back.
            if force or now - self._last_3d >= max(0.1, self.scene.cost_s * 3.0):
                self._last_3d = now
                self.scene.update_time(self.t, force=False)

    def _update_curves(self):
        session = self.session
        if session is None or not self.curve_panel.isVisible():
            return
        window = float(self.combo_window.currentData() or 0.0)
        if window <= 0 or window >= session.duration:
            start, stop = 0.0, max(session.duration, 1e-3)
        else:
            start = float(np.clip(self.t - window / 2, 0.0, session.duration - window))
            stop = start + window
        self.curve_panel.set_view(start, stop, self.t)

    def _show_video(self, force=False):
        session = self.session
        if session is None or not self.video_view.isVisible():
            return
        video = session.video
        if video is None or not video.count:
            self.video_view.show_images([], "This session has no video")
            return
        index = video.index_at(self.t)
        eye = self.combo_eye.currentData() if "right" in video.eyes else "left"
        key = (index, eye)
        if key == self._shown_frame and not force:
            return
        self._shown_frame = key
        eyes = ("left", "right") if eye == "both" else (eye or "left",)
        self.video_view.show_images([video.frame(index, name) for name in eyes])

    # ---------------------------------------------------------------- export
    def _default_name(self, suffix):
        meta = self.session.meta
        who = meta.get("participant") or {}
        specimen = str((meta.get("specimen") or {}).get("name", "specimen")).replace(" ", "_")
        person = f"{who.get('last_name', '')}_{who.get('first_name', '')}".strip("_") or "session"
        return str(self.session.folder / f"{person}_{specimen}_session{meta.get('session_number', '')}{suffix}")

    def export_video(self):
        self.pause()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export video", self._default_name(".mp4"), "MP4 video (*.mp4)")
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        start, stop = self._range()
        eye = self.combo_eye.currentData() or "left"
        dialog = QtWidgets.QProgressDialog("Writing video...", "Cancel", 0, 100, self)
        dialog.setWindowModality(Qt.WindowModal)

        def progress(done, total):
            dialog.setValue(int(done * 100 / total))
            QtWidgets.QApplication.processEvents()
            return not dialog.wasCanceled()

        try:
            count = data.export_video(self.session, path, start, stop, eye, progress)
            cancelled = dialog.wasCanceled()
            dialog.close()
            if cancelled:
                pathlib.Path(path).unlink(missing_ok=True)
            else:
                QtWidgets.QMessageBox.information(self, "Export video", f"Wrote {count:,} frames to\n{path}")
        except (OSError, ValueError) as exc:
            dialog.close()
            QtWidgets.QMessageBox.warning(self, "Export video", str(exc))

    def export_csv(self):
        self.pause()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export curves", self._default_name(".csv"), "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        start, stop = self._range()
        keys = self.selected_curves()
        try:
            rows = data.export_csv(self.session, path, start, stop, keys)
            QtWidgets.QMessageBox.information(self, "Export curves",
                                              f"Wrote {len(keys)} curves, {rows:,} rows at 100 Hz, to\n{path}")
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Export curves", str(exc))

    def shutdown(self):
        if self._shut_down:
            return
        self._shut_down = True
        self.timer.stop()
        self.scene.shutdown()
        if self.session is not None:
            self.session.close()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


def default_base():
    setup = REPO / "scripts" / "study_gui" / "gui_setup.yaml"
    try:
        return yaml.safe_load(setup.read_text())["recording_base_path"]
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return str(pathlib.Path.home())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", nargs="?", help="Session folder to open")
    parser.add_argument("--base", default=None, help="Recording folder holding expert_drillings[_dev]")
    parser.add_argument("--grab", help=argparse.SUPPRESS)  # save a screenshot after loading, then exit
    parser.add_argument("--at", type=float, default=0.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("FIVRS Session Review")
    app.setApplicationDisplayName("FIVRS Session Review")
    app.setDesktopFileName("fivrs-session-review")
    app.setWindowIcon(QtGui.QIcon(str(HERE / "assets" / "session_review_icon.png")))
    window = ReviewWindow(args.base or default_base())
    screen = app.primaryScreen().availableGeometry()
    window.resize(min(1680, screen.width() - 60), min(1020, screen.height() - 60))
    window.show()
    if args.session:
        QtCore.QTimer.singleShot(50, lambda: window.load(args.session))
    if args.grab:
        def grab():
            window.raise_()
            window.activateWindow()
            window.seek(args.at)
            app.processEvents()
            QtCore.QTimer.singleShot(700, finish)

        def finish():
            app.primaryScreen().grabWindow(window.winId()).save(args.grab)
            window.close()

        QtCore.QTimer.singleShot(1800, grab)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
