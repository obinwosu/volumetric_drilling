"""The Review tab's 3D view: the case-review site's GPU ray caster, in a native OpenGL window.

The volume goes to the GPU once, with each voxel's structure and its removal number. Moving in time
changes one number, so nothing is re-meshed. The shaders are read from the site's app.js, so the
two views cannot drift apart. OpenGL is called through ctypes: the launcher's Python has no GL package.
"""
from __future__ import annotations

import ctypes
import math
import pathlib
import re
import time
from ctypes import POINTER, c_char, c_char_p, c_float, c_int, c_size_t, c_ubyte, c_uint

import numpy as np
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal

WEB_APP = pathlib.Path(__file__).resolve().parent.parent / "remote_review" / "web" / "app.js"
BRICK = 8
NEVER = 0xFFFFFFFF
REST_SCALE = 2.0  # supersampling once the picture is still
REST_AFTER_S = 0.18

GL_TEXTURE_2D, GL_TEXTURE_3D, GL_TEXTURE0 = 0x0DE1, 0x806F, 0x84C0
GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_NEAREST, GL_LINEAR = 0x2800, 0x2801, 0x2600, 0x2601
GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE = 0x2802, 0x2803, 0x8072, 0x812F
GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE, GL_UNSIGNED_INT, GL_FLOAT = 0x0CF5, 0x1401, 0x1405, 0x1406
GL_RED_INTEGER, GL_R8UI, GL_R32UI, GL_RGBA, GL_RGBA8 = 0x8D94, 0x8232, 0x8236, 0x1908, 0x8058
GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS, GL_LINK_STATUS = 0x8B31, 0x8B30, 0x8B81, 0x8B82
GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_TRIANGLES, GL_LINE_STRIP = 0x8892, 0x88E4, 0x0004, 0x0003
GL_FRAMEBUFFER, GL_READ_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER, GL_RENDERBUFFER = 0x8D40, 0x8CA8, 0x8CA9, 0x8D41
GL_COLOR_ATTACHMENT0, GL_FRAMEBUFFER_COMPLETE, GL_COLOR_BUFFER_BIT = 0x8CE0, 0x8CD5, 0x4000

_I, _U, _F, _P = c_int, c_uint, c_float, ctypes.c_void_p
_SIGNATURES = {
    "glGenTextures": (None, (_I, POINTER(_U))), "glBindTexture": (None, (_U, _U)), "glActiveTexture": (None, (_U,)),
    "glDeleteTextures": (None, (_I, POINTER(_U))), "glTexParameteri": (None, (_U, _U, _I)),
    "glPixelStorei": (None, (_U, _I)),
    "glTexImage2D": (None, (_U, _I, _I, _I, _I, _I, _U, _U, _P)),
    "glTexImage3D": (None, (_U, _I, _I, _I, _I, _I, _I, _U, _U, _P)),
    "glTexSubImage3D": (None, (_U, _I, _I, _I, _I, _I, _I, _I, _U, _U, _P)),
    "glCreateShader": (_U, (_U,)), "glShaderSource": (None, (_U, _I, POINTER(c_char_p), POINTER(_I))),
    "glCompileShader": (None, (_U,)), "glGetShaderiv": (None, (_U, _U, POINTER(_I))),
    "glGetShaderInfoLog": (None, (_U, _I, POINTER(_I), POINTER(c_char))), "glCreateProgram": (_U, ()),
    "glAttachShader": (None, (_U, _U)), "glLinkProgram": (None, (_U,)), "glGetProgramiv": (None, (_U, _U, POINTER(_I))),
    "glGetProgramInfoLog": (None, (_U, _I, POINTER(_I), POINTER(c_char))), "glUseProgram": (None, (_U,)),
    "glGetUniformLocation": (_I, (_U, c_char_p)), "glGetAttribLocation": (_I, (_U, c_char_p)),
    "glUniform1i": (None, (_I, _I)), "glUniform1ui": (None, (_I, _U)), "glUniform1f": (None, (_I, _F)),
    "glUniform3i": (None, (_I, _I, _I, _I)), "glUniform3f": (None, (_I, _F, _F, _F)),
    "glUniform4f": (None, (_I, _F, _F, _F, _F)), "glGenVertexArrays": (None, (_I, POINTER(_U))),
    "glBindVertexArray": (None, (_U,)), "glGenBuffers": (None, (_I, POINTER(_U))), "glBindBuffer": (None, (_U, _U)),
    "glBufferData": (None, (_U, c_size_t, _P, _U)), "glEnableVertexAttribArray": (None, (_U,)),
    "glDisableVertexAttribArray": (None, (_U,)), "glVertexAttribPointer": (None, (_U, _I, _U, c_ubyte, _I, _P)),
    "glDrawArrays": (None, (_U, _I, _I)), "glViewport": (None, (_I, _I, _I, _I)),
    "glClearColor": (None, (_F, _F, _F, _F)), "glClear": (None, (_U,)), "glGenFramebuffers": (None, (_I, POINTER(_U))),
    "glBindFramebuffer": (None, (_U, _U)), "glGenRenderbuffers": (None, (_I, POINTER(_U))),
    "glBindRenderbuffer": (None, (_U, _U)), "glRenderbufferStorage": (None, (_U, _U, _I, _I)),
    "glFramebufferRenderbuffer": (None, (_U, _U, _U, _U)), "glCheckFramebufferStatus": (_U, (_U,)),
    "glBlitFramebuffer": (None, (_I, _I, _I, _I, _I, _I, _I, _I, _U, _U)), "glFinish": (None, ()),
    "glGetError": (_U, ()),
}


def web_shaders() -> dict:
    """The site's shaders, with the WebGL version line swapped for desktop OpenGL."""
    text = WEB_APP.read_text()
    found = {}
    for name in ("VERT", "FRAG", "LINE_VERT", "LINE_FRAG"):
        match = re.search(rf"const {name} = `(.*?)`;", text, re.S)
        if match is None:
            raise RuntimeError(f"Shader {name} not found in app.js")
        found[name] = match[1].replace("${BRICK}", str(BRICK)).replace("#version 300 es", "#version 410 core")
    return found


def brick_maxima(values, mask):
    """Highest removal number per 8-voxel brick among `mask`: lets rays skip empty space."""
    data = np.where(mask, values, 0).astype(np.uint32)
    data = np.pad(data, [(0, -n % BRICK) for n in data.shape])
    nx, ny, nz = (n // BRICK for n in data.shape)
    return data.reshape(nx, BRICK, ny, BRICK, nz, BRICK).max(axis=(1, 3, 5))


class VolumeFrame:
    """Conversions between the world and the shader's voxel grid (voxel i is centred at i + 0.5)."""

    def __init__(self, volume, shape):
        self.rotation = np.asarray(volume["R_world_volume"], float)
        self.position = np.asarray(volume["p_world_volume"], float)
        self.dims = np.asarray(volume["dimensions_sim"], float)
        self.shape = np.asarray(shape, float)
        self.voxel = float(self.dims[0] / self.shape[0])

    def point(self, world):
        local = (np.asarray(world, float) - self.position) @ self.rotation
        return (local + self.dims / 2.0) / self.dims * self.shape + 0.5

    def direction(self, world):
        return np.asarray(world, float) @ self.rotation


def _unit(v):
    return np.asarray(v, float) / max(float(np.linalg.norm(v)), 1e-9)


def _turn(v, axis, angle):
    return v * math.cos(angle) + np.cross(axis, v) * math.sin(angle) + axis * float(np.dot(axis, v)) * (1 - math.cos(angle))


class SceneWindow(QtGui.QOpenGLWindow):
    interacted = pyqtSignal()    # the user took the camera
    released = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self):
        super().__init__(QtGui.QOpenGLWindow.NoPartialUpdate)
        surface = QtGui.QSurfaceFormat()
        surface.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
        surface.setVersion(4, 1)
        surface.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        surface.setDepthBufferSize(0)
        surface.setStencilBufferSize(0)
        surface.setSamples(0)
        surface.setSwapInterval(0)  # a monitor refresh must never hold up the video
        self.setFormat(surface)
        self.gl = None
        self.error = ""
        self.ready = False
        self._pending_volume = None
        self._pending_palette = None
        self._pending_path = None
        self._textures = {}
        self._size = (1, 1, 1)
        self._fbo = self._color = 0
        self._fbo_size = (0, 0)
        self.now = 1
        self.tip = np.zeros(3)
        self.axis = np.array([1.0, 0, 0])
        self.burr_radius, self.shaft_radius, self.shaft_length = 1.0, 0.4, 100.0
        self.show_removed = False
        self.path_count = 0
        self.center = np.zeros(3)
        self.forward, self.up, self.right = np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
        self.distance = 100.0
        self.fov = math.radians(30)
        self._last_motion = 0.0
        self._press = None
        self.paint_ms = 0.0
        self._rest = QtCore.QTimer(self)
        self._rest.setSingleShot(True)
        self._rest.timeout.connect(self.update)

    # ---- what to draw ----
    def set_volume(self, labels, values):
        self._size = tuple(int(n) for n in labels.shape)
        self._pending_volume = (labels, values)
        self.ready = False
        self.update()

    def set_palette(self, rgba, show_removed):
        self.show_removed = bool(show_removed)
        self._pending_palette = np.asarray(rgba, np.uint8)
        self.update()

    def set_path(self, points):
        self._pending_path = np.asarray(points, np.float32)
        self.update()

    def set_frame(self, now, tip, axis, burr_radius, shaft_radius, shaft_length, path_count):
        self.now = max(int(now), 1)
        self.tip, self.axis = np.asarray(tip, float), _unit(axis)
        self.burr_radius, self.shaft_radius, self.shaft_length = burr_radius, shaft_radius, shaft_length
        self.path_count = int(path_count)
        self._moved()

    def _moved(self):
        self._last_motion = time.perf_counter()
        self.update()

    def set_camera(self, eye, forward, up, fov):
        self.forward = _unit(forward)
        self.up = _unit(np.asarray(up, float) - self.forward * float(np.dot(up, self.forward)))
        self.right = _unit(np.cross(self.forward, self.up))
        middle = np.asarray(self._size, float) / 2.0
        self.distance = max(float(np.dot(middle - eye, self.forward)), 1.0)
        self.center = np.asarray(eye, float) + self.forward * self.distance
        self.fov = float(fov)

    def fit(self):
        self.center = np.asarray(self._size, float) / 2.0
        self.fov = math.radians(30)
        self.distance = float(np.linalg.norm(self._size)) * 0.5 / math.tan(self.fov / 2) * 0.92
        self._moved()

    # ---- mouse: left turns, Shift + left or middle pans, right drag or wheel zooms ----
    def mousePressEvent(self, event):
        self._press = event.pos()
        self.interacted.emit()

    def mouseReleaseEvent(self, _event):
        self._press = None
        self.released.emit()

    def mouseMoveEvent(self, event):
        if self._press is None:
            return
        dx, dy = event.x() - self._press.x(), event.y() - self._press.y()
        self._press = event.pos()
        buttons = event.buttons()
        if buttons & Qt.RightButton:
            self._zoom(math.exp(dy * 0.006))
        elif buttons & Qt.MiddleButton or event.modifiers() & Qt.ShiftModifier:
            k = 2 * self.distance * math.tan(self.fov / 2) / max(self.height(), 1)
            self.center = self.center - self.right * dx * k + self.up * dy * k
        else:
            a, b = -dx * 0.006, -dy * 0.006
            self.forward, self.right = _turn(self.forward, self.up, a), _turn(self.right, self.up, a)
            self.forward, self.up = _unit(_turn(self.forward, self.right, b)), _unit(_turn(self.up, self.right, b))
            self.right = _unit(np.cross(self.forward, self.up))
        self._moved()

    def wheelEvent(self, event):
        self.interacted.emit()
        self._zoom(math.exp(-event.angleDelta().y() * 0.001))
        self._moved()

    def _zoom(self, factor):
        self.distance = min(max(self.distance * factor, 6.0), float(np.linalg.norm(self._size)) * 6.0)

    # ---- OpenGL ----
    def initializeGL(self):
        try:
            context = self.context()
            if tuple(context.format().version()) < (4, 1):
                raise RuntimeError("The 3D view needs OpenGL 4.1.")
            library = ctypes.CDLL("libGL.so.1")
            table = {}
            for name, (result, arguments) in _SIGNATURES.items():
                address = context.getProcAddress(name.encode())
                function = ctypes.CFUNCTYPE(result, *arguments)(int(address)) if address else None
                if function is None:
                    function = getattr(library, name)
                    function.restype, function.argtypes = result, arguments
                table[name[2:]] = function
            self.gl = type("GL", (), {k: staticmethod(v) for k, v in table.items()})
            shaders = web_shaders()
            self.volume_program = self._program(shaders["VERT"], shaders["FRAG"])
            self.line_program = self._program(shaders["LINE_VERT"], shaders["LINE_FRAG"])
            self._vao = self._gen(self.gl.GenVertexArrays)
            self._path_buffer = self._gen(self.gl.GenBuffers)
            self.gl.BindVertexArray(self._vao)
            for unit, target in ((0, GL_TEXTURE_3D), (1, GL_TEXTURE_3D), (2, GL_TEXTURE_2D), (3, GL_TEXTURE_3D), (4, GL_TEXTURE_3D)):
                self._textures[unit] = self._new_texture(unit, target)
        except Exception as exc:  # the review app must survive a machine without the right OpenGL
            self.error = str(exc)
            self.gl = None
            self.failed.emit(self.error)

    @staticmethod
    def _gen(function):
        value = c_uint(0)
        function(1, ctypes.byref(value))
        return value.value

    def _new_texture(self, unit, target):
        gl = self.gl
        texture = self._gen(gl.GenTextures)
        gl.ActiveTexture(GL_TEXTURE0 + unit)
        gl.BindTexture(target, texture)
        for parameter in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
            gl.TexParameteri(target, parameter, GL_NEAREST)
        for parameter in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R):
            gl.TexParameteri(target, parameter, GL_CLAMP_TO_EDGE)
        return texture

    def _program(self, vertex, fragment):
        gl = self.gl
        program = gl.CreateProgram()
        for kind, source in ((GL_VERTEX_SHADER, vertex), (GL_FRAGMENT_SHADER, fragment)):
            shader = gl.CreateShader(kind)
            text = c_char_p(source.encode())
            gl.ShaderSource(shader, 1, ctypes.byref(text), None)
            gl.CompileShader(shader)
            status = c_int(0)
            gl.GetShaderiv(shader, GL_COMPILE_STATUS, ctypes.byref(status))
            if not status.value:
                log = ctypes.create_string_buffer(4096)
                gl.GetShaderInfoLog(shader, 4096, None, log)
                raise RuntimeError(log.value.decode(errors="replace"))
            gl.AttachShader(program, shader)
        gl.LinkProgram(program)
        status = c_int(0)
        gl.GetProgramiv(program, GL_LINK_STATUS, ctypes.byref(status))
        if not status.value:
            log = ctypes.create_string_buffer(4096)
            gl.GetProgramInfoLog(program, 4096, None, log)
            raise RuntimeError(log.value.decode(errors="replace"))
        uniforms = {}
        for name in ("uLabels", "uTimes", "uBricks0", "uBricks1", "uPalette", "uSize", "uNow", "uLimit0", "uShowRemoved",
                     "uShowDrill", "uEye", "uRight", "uUp", "uForward", "uTip", "uAxis", "uTan", "uAspect", "uBurrR",
                     "uShaftR", "uShaftLen", "uScale", "uColor"):
            uniforms[name] = gl.GetUniformLocation(program, name.encode())
        return {"id": program, "u": uniforms}

    def _upload_volume(self):
        gl = self.gl
        labels, values = self._pending_volume
        self._pending_volume = None
        self._labels, self._values = labels, values
        nx, ny, nz = labels.shape
        gl.PixelStorei(GL_UNPACK_ALIGNMENT, 1)
        for unit, data, internal, kind in ((0, labels, GL_R8UI, GL_UNSIGNED_BYTE), (1, values, GL_R32UI, GL_UNSIGNED_INT)):
            payload = data.tobytes(order="F")
            gl.ActiveTexture(GL_TEXTURE0 + unit)
            gl.BindTexture(GL_TEXTURE_3D, self._textures[unit])
            gl.TexImage3D(GL_TEXTURE_3D, 0, internal, nx, ny, nz, 0, GL_RED_INTEGER, kind, payload)
        self._brick_key = None

    def _upload_palette(self):
        gl = self.gl
        palette = self._pending_palette
        self._pending_palette = None
        self._palette_alpha = palette[:, 3].copy()
        gl.ActiveTexture(GL_TEXTURE0 + 2)
        gl.BindTexture(GL_TEXTURE_2D, self._textures[2])
        gl.TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, palette.tobytes())

    def _update_bricks(self):
        alpha = self._palette_alpha
        key = (np.where(alpha == 0, 0, np.where(alpha >= 254, 2, 1)).astype(np.uint8).tobytes(), self.show_removed)
        if key == self._brick_key:
            return
        self._brick_key = key
        gl = self.gl
        shown = alpha[self._labels]
        any_voxel = self._labels > 0 if self.show_removed else shown > 0
        for unit, mask in ((3, any_voxel), (4, shown >= 254)):
            bricks = brick_maxima(self._values, mask)
            bx, by, bz = bricks.shape
            gl.ActiveTexture(GL_TEXTURE0 + unit)
            gl.BindTexture(GL_TEXTURE_3D, self._textures[unit])
            gl.TexImage3D(GL_TEXTURE_3D, 0, GL_R32UI, bx, by, bz, 0, GL_RED_INTEGER, GL_UNSIGNED_INT,
                          bricks.tobytes(order="F"))

    def _ensure_target(self, width, height):
        gl = self.gl
        if not self._fbo:
            self._fbo, self._color = self._gen(gl.GenFramebuffers), self._gen(gl.GenRenderbuffers)
        if self._fbo_size != (width, height):
            gl.BindRenderbuffer(GL_RENDERBUFFER, self._color)
            gl.RenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height)
            gl.BindFramebuffer(GL_FRAMEBUFFER, self._fbo)
            gl.FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self._color)
            self._fbo_size = (width, height)

    def _camera(self, program, aspect):
        gl, u = self.gl, program["u"]
        eye = self.center - self.forward * self.distance
        for name, vector in (("uEye", eye), ("uRight", self.right), ("uUp", self.up), ("uForward", self.forward)):
            gl.Uniform3f(u[name], *(float(v) for v in vector))
        gl.Uniform1f(u["uTan"], math.tan(self.fov / 2))
        gl.Uniform1f(u["uAspect"], aspect)

    def paintGL(self):
        if self.gl is None:
            return
        gl = self.gl
        began = time.perf_counter()
        target = self.defaultFramebufferObject()
        ratio = self.devicePixelRatio()
        out_w, out_h = max(int(self.width() * ratio), 2), max(int(self.height() * ratio), 2)
        gl.BindVertexArray(self._vao)
        if self._pending_volume is not None:
            self._upload_volume()
        if self._pending_palette is not None:
            self._upload_palette()
        if self._pending_path is not None:
            data = self._pending_path.tobytes()
            gl.BindBuffer(GL_ARRAY_BUFFER, self._path_buffer)
            gl.BufferData(GL_ARRAY_BUFFER, len(data), data, GL_STATIC_DRAW)
            self._pending_path = None
        if not hasattr(self, "_labels") or not hasattr(self, "_palette_alpha"):
            gl.BindFramebuffer(GL_FRAMEBUFFER, target)
            gl.ClearColor(0.02, 0.05, 0.08, 1.0)
            gl.Clear(GL_COLOR_BUFFER_BIT)
            return
        self._update_bricks()
        self.ready = True
        moving = time.perf_counter() - self._last_motion < REST_AFTER_S
        scale = 1.0 if moving else REST_SCALE
        width, height = int(out_w * scale), int(out_h * scale)
        self._ensure_target(width, height)
        gl.BindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        gl.Viewport(0, 0, width, height)
        aspect = width / height
        program, u = self.volume_program, self.volume_program["u"]
        gl.UseProgram(program["id"])
        for name, unit in (("uLabels", 0), ("uTimes", 1), ("uPalette", 2), ("uBricks0", 3), ("uBricks1", 4)):
            gl.Uniform1i(u[name], unit)
        gl.Uniform3i(u["uSize"], *self._size)
        now = min(self.now, NEVER - 1)
        gl.Uniform1ui(u["uNow"], now)
        gl.Uniform1ui(u["uLimit0"], 0 if self.show_removed else now)
        gl.Uniform1i(u["uShowRemoved"], int(self.show_removed))
        gl.Uniform1i(u["uShowDrill"], 1)
        gl.Uniform3f(u["uTip"], *(float(v) for v in self.tip))
        gl.Uniform3f(u["uAxis"], *(float(v) for v in self.axis))
        gl.Uniform1f(u["uBurrR"], self.burr_radius)
        gl.Uniform1f(u["uShaftR"], self.shaft_radius)
        gl.Uniform1f(u["uShaftLen"], self.shaft_length)
        self._camera(program, aspect)
        gl.DrawArrays(GL_TRIANGLES, 0, 3)
        if self.path_count > 1:
            lines = self.line_program
            gl.UseProgram(lines["id"])
            self._camera(lines, aspect)
            gl.Uniform1f(lines["u"]["uScale"], 1.0)
            gl.Uniform4f(lines["u"]["uColor"], 0.3, 0.85, 1.0, 1.0)
            gl.BindBuffer(GL_ARRAY_BUFFER, self._path_buffer)
            attribute = gl.GetAttribLocation(lines["id"], b"point")
            gl.EnableVertexAttribArray(attribute)
            gl.VertexAttribPointer(attribute, 3, GL_FLOAT, 0, 0, None)
            gl.DrawArrays(GL_LINE_STRIP, 0, self.path_count)
            gl.DisableVertexAttribArray(attribute)
        gl.BindFramebuffer(GL_READ_FRAMEBUFFER, self._fbo)
        gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, target)
        gl.BlitFramebuffer(0, 0, width, height, 0, 0, out_w, out_h, GL_COLOR_BUFFER_BIT, GL_LINEAR)
        gl.BindFramebuffer(GL_FRAMEBUFFER, target)
        self.paint_ms = (time.perf_counter() - began) * 1000.0
        if moving:
            self._rest.start(int(REST_AFTER_S * 1000) + 20)
