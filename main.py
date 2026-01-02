# © 2026 Paul Hiret
# Licensed under CC BY-NC 4.0

import os
import sys
import time
import struct
import ctypes as C
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import webbrowser
import numpy as np
import h5py
import cv2

from PySide6 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

def resource_dir() -> Path:
    # In PyInstaller onefile, resources get unpacked to sys._MEIPASS
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

SIGMA = 5.670374419e-8
DEFAULT_FPS = 27
DEFAULT_T_RANGE = (0, 250)  # you confirmed this fixes your init


# ============================================================
# Portable paths (relative to this main.py)
# ============================================================
BASE_DIR = resource_dir()
SDK_DIR = BASE_DIR / "sdk" / "x64"
DLL_NAME = "libirimager.dll"
CONFIG_XML = BASE_DIR / "sdk" / "generic.xml"

ICON_PATH = BASE_DIR / "icon.ico"
SPLASH_PATH = BASE_DIR / "splash.png"

APP_NAME = "Optris_IR_ESCA"
APP_VERSION = "1.0.0"
BUILD_DATE = datetime.now().strftime("%Y-%m-%d")
MANUAL_URL = (
    "https://www.notion.so/paulhiret/"
    "IRCamera-Imager-UB-User-Procedure-2735aa0ff72f80b58024c86f0eb73271"
)



# ============================================================
# Helpers
# ============================================================
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _check_paths():
    print("Python bitness:", struct.calcsize("P") * 8, "bit")
    dll_path = SDK_DIR / DLL_NAME
    if not dll_path.exists():
        raise FileNotFoundError(f"DLL not found:\n{dll_path}")
    if not CONFIG_XML.exists():
        raise FileNotFoundError(f"Config XML not found:\n{CONFIG_XML}")
    return str(dll_path), str(CONFIG_XML)


def planck_integrated_radiance_75_14um(T_kelvin: float, n=2000) -> float:
    """Integrate Planck spectral radiance B_lambda over 7.5–14 µm."""
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23

    lam1 = 7.5e-6
    lam2 = 14e-6

    wl = np.linspace(lam1, lam2, n)
    c1 = 2 * h * c**2
    c2 = h * c / k

    expo = np.exp(c2 / (wl * T_kelvin))
    B = (c1 / (wl**5)) * (1.0 / (expo - 1.0))
    return float(np.trapz(B, wl))

def app_roaming_dir(app_name: str = APP_NAME) -> Path:
    # Windows roaming AppData (portable, user-specific)
    base = os.environ.get("APPDATA", None)
    if base:
        return Path(base) / app_name
    # fallback (shouldn't happen on Windows)
    return Path.home() / f".{app_name}"

def ensure_log_dir() -> Path:
    d = app_roaming_dir(APP_NAME) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d

def log_file_path() -> Path:
    # daily log file
    return ensure_log_dir() / f"{APP_NAME.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.log"


# ============================================================
# Data
# ============================================================
@dataclass
class FrameBundle:
    raw_u16: Optional[np.ndarray]        # (H,W) uint16
    temp_c_wo_emi: np.ndarray            # (H,W) float64


# ============================================================
# Optris Camera Wrapper (ctypes)
# ============================================================
class OptrisIR:
    def __init__(self, dll_path: str, config_xml: str):
        self.dll_path = dll_path
        self.config_path_str = os.path.abspath(config_xml)
        self.config_path = self.config_path_str.encode("utf-8")

        # Make sure Windows DLL search includes our folder (PyInstaller-friendly)
        try:
            os.add_dll_directory(str(Path(dll_path).parent))
        except Exception:
            pass

        # WinDLL is typically correct for this SDK on Windows
        self.lib = C.WinDLL(dll_path)

        self.connected = False
        self.w_thm = 0
        self.h_thm = 0
        self.thm_buf = None

        self.emissivity = 1.0
        self.transmission = 1.0

        self._has_flag = False
        self._bind_prototypes()

    def _try_get_func(self, name: str):
        try:
            return getattr(self.lib, name)
        except AttributeError:
            return None

    def _bind_prototypes(self):
        self.lib.evo_irimager_usb_init.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p]
        self.lib.evo_irimager_usb_init.restype = C.c_int

        self.lib.evo_irimager_terminate.argtypes = []
        self.lib.evo_irimager_terminate.restype = None

        self.lib.evo_irimager_get_thermal_image_size.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int)]
        self.lib.evo_irimager_get_thermal_image_size.restype = None

        self.lib.evo_irimager_set_temperature_range.argtypes = [C.c_int, C.c_int]
        self.lib.evo_irimager_set_temperature_range.restype = None

        # thermal getters (different SDK builds)
        self._thermal_getters = []
        for name in (
            "evo_irimager_get_thermal_image",
            "evo_irimager_get_thermal_image_byref",
            "evo_irimager_get_thermal_image_by_ref",
        ):
            fn = self._try_get_func(name)
            if fn is None:
                continue
            fn.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_uint16)]
            fn.restype = C.c_int
            self._thermal_getters.append((name, fn))

        if not self._thermal_getters:
            raise RuntimeError("No thermal getter found in DLL.")

        flag_func = self._try_get_func("evo_irimager_trigger_shutter_flag")
        if flag_func is not None:
            flag_func.argtypes = []
            flag_func.restype = C.c_int
            self._has_flag = True

    def connect(self):
        xml_dir = os.path.dirname(self.config_path_str)
        old_cwd = os.getcwd()

        formats_def = os.path.join(xml_dir, "formats.def")
        if not os.path.exists(formats_def):
            raise FileNotFoundError(f"formats.def missing:\n{formats_def}")

        try:
            # Important: XML uses relative paths -> set CWD to XML folder
            os.chdir(xml_dir)

            # IMPORTANT: pass NULL pointers, not b""
            rc = self.lib.evo_irimager_usb_init(self.config_path, None, None)
            if rc != 0:
                raise RuntimeError(f"evo_irimager_usb_init failed rc={rc}")

            w = C.c_int(0)
            h = C.c_int(0)
            self.lib.evo_irimager_get_thermal_image_size(C.byref(w), C.byref(h))
            if w.value <= 0 or h.value <= 0:
                raise RuntimeError(f"invalid thermal size w={w.value} h={h.value}")

            self.w_thm, self.h_thm = w.value, h.value
            n = self.w_thm * self.h_thm
            self.thm_buf = (C.c_uint16 * n)()
            self.connected = True

            # choose best getter once
            self._best_getter = None
            self._select_best_getter()

        finally:
            os.chdir(old_cwd)

    def _select_best_getter(self):
        """Pick first working getter; prefer non-constant frames."""
        w = C.c_int(self.w_thm)
        h = C.c_int(self.h_thm)
        buf_ptr = C.cast(self.thm_buf, C.POINTER(C.c_uint16))

        best = None
        for name, fn in self._thermal_getters:
            ok = False
            for _ in range(8):
                rc = fn(C.byref(w), C.byref(h), buf_ptr)
                if rc != 0:
                    break
                raw = self._buffer_to_numpy()
                if np.unique(raw).size > 1:
                    best = (name, fn)
                    ok = True
                    break
                time.sleep(0.02)
            if ok:
                break

            if best is None and rc == 0:
                best = (name, fn)

        if best is None:
            raise RuntimeError("All thermal getters failed.")
        self._best_getter = best

    def terminate(self):
        if self.connected:
            try:
                self.lib.evo_irimager_terminate()
            finally:
                self.connected = False

    def set_temperature_range(self, tmin: int, tmax: int):
        self.lib.evo_irimager_set_temperature_range(int(tmin), int(tmax))

    def trigger_shutter_flag(self):
        if not self.connected:
            raise RuntimeError("not connected")
        if not self._has_flag:
            raise RuntimeError("trigger_shutter_flag not available in this DLL")
        rc = self.lib.evo_irimager_trigger_shutter_flag()
        if rc != 0:
            raise RuntimeError(f"trigger_shutter_flag failed rc={rc}")

    def _buffer_to_numpy(self) -> np.ndarray:
        n = self.w_thm * self.h_thm
        arr = np.ctypeslib.as_array(self.thm_buf, shape=(n,)).reshape((self.h_thm, self.w_thm))
        return arr.copy()

    def get_thermal_raw_u16(self) -> np.ndarray:
        if not self.connected:
            raise RuntimeError("not connected")

        w = C.c_int(self.w_thm)
        h = C.c_int(self.h_thm)
        buf_ptr = C.cast(self.thm_buf, C.POINTER(C.c_uint16))

        name, fn = self._best_getter
        rc = fn(C.byref(w), C.byref(h), buf_ptr)
        if rc != 0:
            raise RuntimeError(f"{name} failed rc={rc}")

        return self._buffer_to_numpy()

    @staticmethod
    def raw_to_celsius_wo_emi(raw_u16: np.ndarray) -> np.ndarray:
        # matches your MATLAB: 0.1*(raw-1000)
        return 0.1 * (raw_u16.astype(np.float64) - 1000.0)

    def capture_one(self) -> FrameBundle:
        raw = self.get_thermal_raw_u16()
        temp_c_wo = self.raw_to_celsius_wo_emi(raw)
        return FrameBundle(raw_u16=raw, temp_c_wo_emi=temp_c_wo)


# ============================================================
# Matplotlib image widget (toolbar + ROI + cursor readout)
# ============================================================
class MplImageWidget(QtWidgets.QWidget):
    roiChanged = QtCore.Signal(object)  # (x0,y0,x1,y1) or None
    cursorValue = QtCore.Signal(object) # dict: {x,y,val,raw}

    def __init__(self, parent=None):
        super().__init__(parent)

        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # layout
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self.toolbar)
        v.addWidget(self.canvas)

        # axes
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()

        self.im = None
        self.cbar = None

        self._data = None
        self._raw = None
        self._cmap = "turbo"
        self._vmin = None
        self._vmax = None

        # ROI selector
        self._selector = RectangleSelector(
            self.ax, self._on_select,
            useblit=True,
            button=[1],  # left click
            interactive=True
        )
        self._selector.set_active(True)

        # cursor readout
        self._cid_move = self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    def set_cmap(self, cmap: str):
        self._cmap = cmap
        if self.im is not None:
            self.im.set_cmap(cmap)
            self.canvas.draw_idle()

    def set_scale(self, vmin: Optional[float], vmax: Optional[float]):
        self._vmin = vmin
        self._vmax = vmax
        if self.im is not None and vmin is not None and vmax is not None:
            self.im.set_clim(vmin, vmax)
            self.canvas.draw_idle()

    def clear_roi(self):
        # RectangleSelector doesn't have "clear", so we just emit None
        self.roiChanged.emit(None)

    def set_image(self, data2d: np.ndarray, raw_u16: Optional[np.ndarray], label: str,
                  vmin: Optional[float], vmax: Optional[float], cmap: str):
        self._data = data2d
        self._raw = raw_u16
        self._cmap = cmap
        self._vmin = vmin
        self._vmax = vmax

        # First time: create imshow + colorbar once
        if self.im is None:
            self.ax.clear()
            self.ax.set_axis_off()

            self.im = self.ax.imshow(
                data2d, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper"
            )

            # Create ONE persistent colorbar
            self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
            self.cbar.set_label(label)

        else:
            # Update existing image (fast, no new axes)
            self.im.set_data(data2d)
            self.im.set_cmap(cmap)

            if vmin is not None and vmax is not None:
                self.im.set_clim(vmin, vmax)

            # Update colorbar label + ticks
            if self.cbar is not None:
                self.cbar.set_label(label)
                self.cbar.update_normal(self.im)

        self.canvas.draw_idle()

        # Recreate selector bound to current axes (avoid stacking selectors)
        try:
            self._selector.set_active(False)
        except Exception:
            pass

        self._selector = RectangleSelector(
            self.ax, self._on_select,
            useblit=True,
            button=[1],
            interactive=True
        )
        self._selector.set_active(True)

    def _on_select(self, eclick, erelease):
        if self._data is None:
            self.roiChanged.emit(None)
            return
        if eclick.xdata is None or erelease.xdata is None:
            self.roiChanged.emit(None)
            return

        x0 = int(np.floor(min(eclick.xdata, erelease.xdata)))
        x1 = int(np.ceil(max(eclick.xdata, erelease.xdata)))
        y0 = int(np.floor(min(eclick.ydata, erelease.ydata)))
        y1 = int(np.ceil(max(eclick.ydata, erelease.ydata)))

        h, w = self._data.shape
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, 0, w))
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, 0, h))

        if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
            self.roiChanged.emit(None)
            return

        self.roiChanged.emit((x0, y0, x1, y1))

    def _on_mouse_move(self, event):
        if self._data is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        h, w = self._data.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        val = float(self._data[y, x])
        raw = None
        if self._raw is not None:
            raw = int(self._raw[y, x])

        self.cursorValue.emit({"x": x, "y": y, "val": val, "raw": raw})


# ============================================================
# Main GUI
# ============================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cam: OptrisIR):
        super().__init__()
        self.log_path = log_file_path()
        self.cam = cam

        self.setWindowTitle("OptrisIR: ESCA")
        if ICON_PATH.exists():
            self.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_tick)

        # state
        self.mode = "Temperature"  # or Radiance
        self.palette = "turbo"     # matplotlib cmap name
        self.scale_auto = True
        self.scale_min = 0.0
        self.scale_max = 1.0
        self.live_enabled = False
        self.recording = False
        self.frames: list[FrameBundle] = []
        self.current_index = 0
        self.last_bundle: Optional[FrameBundle] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None

        # build UI
        self._build_menu()
        self._build_ui()
        self._build_hotkeys()
        self._set_connected(False)

        self.statusBar().showMessage("Ready")


    # ---------------- Menus ----------------
    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        act_open = QtGui.QAction("Open .h5...", self)
        act_save = QtGui.QAction("Save .h5...", self)
        act_export_avi = QtGui.QAction("Export AVI...", self)
        act_save_csv_temp = QtGui.QAction("Save CSV (Temp corrected)...", self)
        act_save_csv_raw = QtGui.QAction("Save CSV (RAW u16)...", self)
        act_save_csv_wo = QtGui.QAction("Save CSV (Temp woEmi)...", self)
        act_save_tiff = QtGui.QAction("Save TIFF (single frame)...", self)
        act_exit = QtGui.QAction("Exit", self)

        file_menu.addAction(act_open)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        file_menu.addAction(act_save_csv_temp)
        file_menu.addAction(act_save_csv_raw)
        file_menu.addAction(act_save_csv_wo)
        file_menu.addAction(act_save_tiff)
        file_menu.addSeparator()
        file_menu.addAction(act_export_avi)
        file_menu.addSeparator()
        file_menu.addAction(act_exit)

        act_open.triggered.connect(self.open_h5)
        act_save.triggered.connect(self.save_h5)
        act_export_avi.triggered.connect(self.export_avi)
        act_save_csv_temp.triggered.connect(self.save_csv_temp)
        act_save_csv_raw.triggered.connect(self.save_csv_raw)
        act_save_csv_wo.triggered.connect(self.save_csv_temp_wo)
        act_save_tiff.triggered.connect(self.save_tiff_single)
        act_exit.triggered.connect(self.close)

        cam_menu = mb.addMenu("Camera")
        act_connect = QtGui.QAction("Connect", self)
        act_disconnect = QtGui.QAction("Disconnect", self)
        act_capture = QtGui.QAction("Capture", self)
        act_live = QtGui.QAction("Start/Stop Live", self)
        act_record = QtGui.QAction("Record toggle", self)

        cam_menu.addAction(act_connect)
        cam_menu.addAction(act_disconnect)
        cam_menu.addSeparator()
        cam_menu.addAction(act_capture)
        cam_menu.addAction(act_live)
        cam_menu.addAction(act_record)

        act_connect.triggered.connect(self.connect_camera)
        act_disconnect.triggered.connect(self.disconnect_camera)
        act_capture.triggered.connect(self.capture)
        act_live.triggered.connect(self.toggle_live)
        act_record.triggered.connect(self.toggle_record)

        view_menu = mb.addMenu("View")
        act_clear_roi = QtGui.QAction("Clear ROI", self)
        act_auto_scale = QtGui.QAction("Auto scale", self)
        act_freeze = QtGui.QAction("Freeze scale from frame", self)
        act_light = QtGui.QAction("Light mode", self, checkable=True)
        act_dark = QtGui.QAction("Dark mode", self, checkable=True)
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        group.addAction(act_light)
        group.addAction(act_dark)
        act_light.setChecked(True)  # default
        act_light.triggered.connect(lambda: self.apply_theme("Light"))
        act_dark.triggered.connect(lambda: self.apply_theme("Dark"))
        view_menu.addAction(act_light)
        view_menu.addAction(act_dark)

        view_menu.addAction(act_clear_roi)
        view_menu.addAction(act_auto_scale)
        view_menu.addAction(act_freeze)

        act_clear_roi.triggered.connect(self.clear_roi)
        act_auto_scale.triggered.connect(lambda: self.chk_auto.setChecked(True))
        act_freeze.triggered.connect(self.freeze_scale_from_frame)

        # =========================
        # Help menu
        # =========================
        help_menu = mb.addMenu("&Help")

        act_manual = QtGui.QAction("&User manual", self)
        act_manual.setShortcut("F1")
        act_manual.setStatusTip("Open the IRCamera Imager UB user procedure")
        act_manual.triggered.connect(self.open_manual)

        act_open_logs = QtGui.QAction("Open &log folder", self)
        act_open_logs.setStatusTip("Open application log directory")
        act_open_logs.triggered.connect(self.open_log_folder)

        act_about = QtGui.QAction("&About", self)
        act_about.triggered.connect(self.show_about)

        help_menu.addAction(act_manual)
        help_menu.addSeparator()
        help_menu.addAction(act_open_logs)
        help_menu.addSeparator()
        help_menu.addAction(act_about)

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QGridLayout(central)
        root.setColumnStretch(0, 3)
        root.setColumnStretch(1, 2)

        # LEFT: Matplotlib image + toolbar
        self.img_widget = MplImageWidget()
        self.img_widget.roiChanged.connect(self.on_roi_changed)
        self.img_widget.cursorValue.connect(self.on_cursor_value)

        # Make LEFT scrollable (as requested)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self.img_widget)
        root.addWidget(left_scroll, 0, 0, 1, 1)

        # slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.on_slider)
        root.addWidget(self.slider, 1, 0, 1, 1)

        # RIGHT panel: controls (also scrollable for small screens)
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_container = QtWidgets.QWidget()
        right_scroll.setWidget(right_container)
        right = QtWidgets.QVBoxLayout(right_container)
        root.addWidget(right_scroll, 0, 1, 2, 1)

        # Connection box
        conn_box = QtWidgets.QGroupBox("Connection")
        conn_layout = QtWidgets.QGridLayout(conn_box)
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.lamp_connected = QtWidgets.QLabel("Disconnected")
        self.lamp_connected.setAlignment(QtCore.Qt.AlignCenter)
        self.lamp_connected.setStyleSheet("background:#cc2222; color:white; font-weight:600; padding:4px; border-radius:6px;")
        self.lamp_recording = QtWidgets.QLabel("Not Recording")
        self.lamp_recording.setAlignment(QtCore.Qt.AlignCenter)
        self.lamp_recording.setStyleSheet("background:#666666; color:white; font-weight:600; padding:4px; border-radius:6px;")
        conn_layout.addWidget(self.btn_connect, 0, 0)
        conn_layout.addWidget(self.btn_disconnect, 0, 1)
        conn_layout.addWidget(self.lamp_connected, 1, 0)
        conn_layout.addWidget(self.lamp_recording, 1, 1)
        right.addWidget(conn_box)

        self.btn_connect.clicked.connect(self.connect_camera)
        self.btn_disconnect.clicked.connect(self.disconnect_camera)

        # Parameters
        param_box = QtWidgets.QGroupBox("Parameters")
        form = QtWidgets.QFormLayout(param_box)

        self.spin_eps = QtWidgets.QDoubleSpinBox()
        self.spin_eps.setRange(0.0, 1.0)
        self.spin_eps.setDecimals(3)
        self.spin_eps.setSingleStep(0.01)
        self.spin_eps.setValue(1.0)

        self.spin_tau = QtWidgets.QDoubleSpinBox()
        self.spin_tau.setRange(0.0, 1.0)
        self.spin_tau.setDecimals(3)
        self.spin_tau.setSingleStep(0.01)
        self.spin_tau.setValue(1.0)

        self.combo_range = QtWidgets.QComboBox()
        self.combo_range.addItems(["(-20, 100)", "(0, 250)", "(150, 900)"])
        self.combo_range.setCurrentText("(0, 250)")

        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["Temperature", "Radiance"])
        self.combo_mode.setCurrentText("Temperature")

        # Matplotlib colormaps (working)
        self.combo_palette = QtWidgets.QComboBox()
        self.combo_palette.addItems([
            "turbo", "inferno", "magma", "plasma", "viridis",
            "jet", "hot", "gray"
        ])
        self.combo_palette.setCurrentText("turbo")

        form.addRow("Emissivity", self.spin_eps)
        form.addRow("Transmissivity", self.spin_tau)
        form.addRow("Temp range", self.combo_range)
        form.addRow("Mode", self.combo_mode)
        form.addRow("Palette", self.combo_palette)
        right.addWidget(param_box)

        self.spin_eps.valueChanged.connect(self.on_params_changed)
        self.spin_tau.valueChanged.connect(self.on_params_changed)
        self.combo_mode.currentTextChanged.connect(self.on_mode_changed)
        self.combo_range.currentTextChanged.connect(self.on_range_changed)
        self.combo_palette.currentTextChanged.connect(self.on_palette_changed)

        # caxis
        scale_box = QtWidgets.QGroupBox("Display scale (caxis)")
        scale_layout = QtWidgets.QGridLayout(scale_box)

        self.chk_auto = QtWidgets.QCheckBox("Auto")
        self.chk_auto.setChecked(True)

        self.spin_cmin = QtWidgets.QDoubleSpinBox()
        self.spin_cmin.setRange(-1e12, 1e12)
        self.spin_cmin.setDecimals(3)
        self.spin_cmin.setValue(0.0)

        self.spin_cmax = QtWidgets.QDoubleSpinBox()
        self.spin_cmax.setRange(-1e12, 1e12)
        self.spin_cmax.setDecimals(3)
        self.spin_cmax.setValue(1.0)

        self.btn_freeze = QtWidgets.QPushButton("Freeze from frame")

        scale_layout.addWidget(self.chk_auto, 0, 0, 1, 2)
        scale_layout.addWidget(QtWidgets.QLabel("Min"), 1, 0)
        scale_layout.addWidget(self.spin_cmin, 1, 1)
        scale_layout.addWidget(QtWidgets.QLabel("Max"), 2, 0)
        scale_layout.addWidget(self.spin_cmax, 2, 1)
        scale_layout.addWidget(self.btn_freeze, 3, 0, 1, 2)

        right.addWidget(scale_box)

        self.chk_auto.toggled.connect(self.on_scale_changed)
        self.spin_cmin.valueChanged.connect(self.on_scale_changed)
        self.spin_cmax.valueChanged.connect(self.on_scale_changed)
        self.btn_freeze.clicked.connect(self.freeze_scale_from_frame)

        self.spin_cmin.setEnabled(False)
        self.spin_cmax.setEnabled(False)

        # Measurements
        meas_box = QtWidgets.QGroupBox("Measurements (Full frame or ROI)")
        meas_form = QtWidgets.QFormLayout(meas_box)
        self.ed_mean_temp = QtWidgets.QLineEdit()
        self.ed_mean_temp.setReadOnly(True)
        self.ed_mean_rad = QtWidgets.QLineEdit()
        self.ed_mean_rad.setReadOnly(True)
        self.lbl_roi = QtWidgets.QLabel("ROI: none")
        self.btn_clear_roi = QtWidgets.QPushButton("Clear ROI")
        self.btn_clear_roi.clicked.connect(self.clear_roi)

        meas_form.addRow("Mean Temp (°C)", self.ed_mean_temp)
        meas_form.addRow("Mean Radiance", self.ed_mean_rad)
        meas_form.addRow(self.lbl_roi, self.btn_clear_roi)
        right.addWidget(meas_box)

        # Emissivity tools
        emi_box = QtWidgets.QGroupBox("Emissivity tools")
        emi_layout = QtWidgets.QGridLayout(emi_box)

        self.spin_tc = QtWidgets.QDoubleSpinBox()
        self.spin_tc.setRange(-50.0, 2000.0)
        self.spin_tc.setValue(25.0)
        self.spin_tc.setDecimals(1)

        self.btn_planck = QtWidgets.QPushButton("Compute emissivity (Planck 7.5–14 µm)")
        self.ed_emi_planck = QtWidgets.QLineEdit()
        self.ed_emi_planck.setReadOnly(True)

        self.spin_bb_rad = QtWidgets.QDoubleSpinBox()
        self.spin_bb_rad.setRange(0.0, 1e25)
        self.spin_bb_rad.setDecimals(6)
        self.spin_bb_rad.setValue(0.0)

        self.btn_bb = QtWidgets.QPushButton("Compute emissivity (Blackbody radiance input)")
        self.ed_emi_bb = QtWidgets.QLineEdit()
        self.ed_emi_bb.setReadOnly(True)

        emi_layout.addWidget(QtWidgets.QLabel("TC Temp (°C):"), 0, 0)
        emi_layout.addWidget(self.spin_tc, 0, 1)
        emi_layout.addWidget(self.btn_planck, 1, 0, 1, 2)
        emi_layout.addWidget(QtWidgets.QLabel("ε (Planck):"), 2, 0)
        emi_layout.addWidget(self.ed_emi_planck, 2, 1)
        emi_layout.addWidget(QtWidgets.QLabel("BB radiance:"), 3, 0)
        emi_layout.addWidget(self.spin_bb_rad, 3, 1)
        emi_layout.addWidget(self.btn_bb, 4, 0, 1, 2)
        emi_layout.addWidget(QtWidgets.QLabel("ε (BB):"), 5, 0)
        emi_layout.addWidget(self.ed_emi_bb, 5, 1)

        self.btn_planck.clicked.connect(self.compute_emissivity_planck)
        self.btn_bb.clicked.connect(self.compute_emissivity_bb)
        right.addWidget(emi_box)

        # Actions
        act_box = QtWidgets.QGroupBox("Actions")
        act_layout = QtWidgets.QGridLayout(act_box)

        self.btn_live = QtWidgets.QPushButton("Start Live")
        self.btn_capture = QtWidgets.QPushButton("Capture")
        self.btn_record = QtWidgets.QPushButton("Record (toggle)")
        self.btn_save = QtWidgets.QPushButton("Save .h5")
        self.btn_open = QtWidgets.QPushButton("Open .h5")
        self.btn_export = QtWidgets.QPushButton("Export AVI")
        self.btn_tiff = QtWidgets.QPushButton("Save TIFF (single frame)")

        self.btn_csv_temp = QtWidgets.QPushButton("Save CSV (Temp corrected)")
        self.btn_csv_raw = QtWidgets.QPushButton("Save CSV (RAW u16)")
        self.btn_csv_wo = QtWidgets.QPushButton("Save CSV (Temp woEmi)")

        act_layout.addWidget(self.btn_live, 0, 0, 1, 2)
        act_layout.addWidget(self.btn_capture, 1, 0)
        act_layout.addWidget(self.btn_record, 1, 1)
        act_layout.addWidget(self.btn_save, 2, 0)
        act_layout.addWidget(self.btn_open, 2, 1)
        act_layout.addWidget(self.btn_export, 3, 0, 1, 2)
        act_layout.addWidget(self.btn_tiff, 4, 0, 1, 2)
        act_layout.addWidget(self.btn_csv_temp, 5, 0, 1, 2)
        act_layout.addWidget(self.btn_csv_raw, 6, 0, 1, 2)
        act_layout.addWidget(self.btn_csv_wo, 7, 0, 1, 2)

        self.btn_live.clicked.connect(self.toggle_live)
        self.btn_capture.clicked.connect(self.capture)
        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_save.clicked.connect(self.save_h5)
        self.btn_open.clicked.connect(self.open_h5)
        self.btn_export.clicked.connect(self.export_avi)
        self.btn_tiff.clicked.connect(self.save_tiff_single)
        self.btn_csv_temp.clicked.connect(self.save_csv_temp)
        self.btn_csv_raw.clicked.connect(self.save_csv_raw)
        self.btn_csv_wo.clicked.connect(self.save_csv_temp_wo)

        right.addWidget(act_box)

        # Log
        log_box = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_box)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        log_layout.addWidget(self.log)
        right.addWidget(log_box)

        right.addStretch(1)

        self._log("App started")
        self._log(f"Log file: {self.log_path}")

    def _build_hotkeys(self):
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+K"), self, activated=self.connect_camera)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, activated=self.disconnect_camera)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_live)
        QtGui.QShortcut(QtGui.QKeySequence("C"), self, activated=self.capture)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, activated=self.toggle_record)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, activated=self.save_h5)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+O"), self, activated=self.open_h5)
        QtGui.QShortcut(QtGui.QKeySequence("E"), self, activated=self.export_avi)
        QtGui.QShortcut(QtGui.QKeySequence("X"), self, activated=self.clear_roi)
        QtGui.QShortcut(QtGui.QKeySequence("F"), self, activated=self.freeze_scale_from_frame)
        QtGui.QShortcut(QtGui.QKeySequence("T"), self, activated=lambda: self.combo_mode.setCurrentText("Temperature"))
        QtGui.QShortcut(QtGui.QKeySequence("G"), self, activated=lambda: self.combo_mode.setCurrentText("Radiance"))

    # ---------------- core helpers ----------------
    def _log(self, msg: str):
        line = f"[{_ts()}] {msg}"
        self.log.append(line)
        self.statusBar().showMessage(msg, 4000)

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _set_connected(self, ok: bool):
        self.btn_connect.setEnabled(not ok)
        self.btn_disconnect.setEnabled(ok)
        self.btn_live.setEnabled(ok)
        self.btn_capture.setEnabled(ok)
        self.btn_record.setEnabled(ok)
        self.combo_range.setEnabled(ok)
        self.spin_eps.setEnabled(ok)
        self.spin_tau.setEnabled(ok)

        if ok:
            self.lamp_connected.setText("Connected")
            self.lamp_connected.setStyleSheet("background:#00cc44; color:white; font-weight:600; padding:4px; border-radius:6px;")
        else:
            self.lamp_connected.setText("Disconnected")
            self.lamp_connected.setStyleSheet("background:#cc2222; color:white; font-weight:600; padding:4px; border-radius:6px;")
            self.lamp_recording.setText("Not Recording")
            self.lamp_recording.setStyleSheet("background:#666666; color:white; font-weight:600; padding:4px; border-radius:6px;")

    def _roi_slice(self, arr: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return arr
        x0, y0, x1, y1 = self.roi
        return arr[y0:y1, x0:x1]

    def _bundle_view(self, bundle: FrameBundle) -> tuple[np.ndarray, np.ndarray]:
        eps = float(self.cam.emissivity)
        tau = float(self.cam.transmission)

        T_k = bundle.temp_c_wo_emi + 273.15
        corr = (eps * tau) ** 0.25
        temp_c = (T_k / corr) - 273.15
        rad = (SIGMA * (T_k ** 4)) / tau
        return temp_c, rad

    def _current_bundle(self) -> Optional[FrameBundle]:
        return self.frames[self.current_index] if self.frames else self.last_bundle

    # ---------------- Cursor readout ----------------
    def on_cursor_value(self, info: dict):
        # show cursor values in status bar (and include raw if present)
        x = info["x"]
        y = info["y"]
        val = info["val"]
        raw = info.get("raw", None)
        if raw is None:
            self.statusBar().showMessage(f"x={x} y={y} value={val:.3f}", 150)
        else:
            self.statusBar().showMessage(f"x={x} y={y} value={val:.3f}  raw_u16={raw}", 150)

    # ---------------- ROI ----------------
    def on_roi_changed(self, roi):
        self.roi = roi
        if roi is None:
            self.lbl_roi.setText("ROI: none")
            self._log("ROI cleared")
        else:
            x0, y0, x1, y1 = roi
            self.lbl_roi.setText(f"ROI: x[{x0}:{x1}] y[{y0}:{y1}]")
            self._log(f"ROI set: x[{x0}:{x1}] y[{y0}:{y1}]")
        self.refresh_display()

    def clear_roi(self):
        self.roi = None
        self.lbl_roi.setText("ROI: none")
        self.img_widget.clear_roi()
        self.refresh_display()

    # ---------------- scale ----------------
    def on_scale_changed(self):
        self.scale_auto = bool(self.chk_auto.isChecked())
        self.spin_cmin.setEnabled(not self.scale_auto)
        self.spin_cmax.setEnabled(not self.scale_auto)

        self.scale_min = float(self.spin_cmin.value())
        self.scale_max = float(self.spin_cmax.value())
        if not self.scale_auto and self.scale_max <= self.scale_min:
            self.scale_max = self.scale_min + 1e-6
            self.spin_cmax.setValue(self.scale_max)

        self.refresh_display()

    def freeze_scale_from_frame(self):
        bundle = self._current_bundle()
        if bundle is None:
            self._log("Freeze scale: no data")
            return
        temp_c, rad = self._bundle_view(bundle)
        img = temp_c if self.mode == "Temperature" else rad
        vmin = float(np.nanmin(img))
        vmax = float(np.nanmax(img))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        self.chk_auto.setChecked(False)
        self.spin_cmin.setValue(vmin)
        self.spin_cmax.setValue(vmax)
        self.on_scale_changed()

    def open_manual(self):
        webbrowser.open(MANUAL_URL)

    def open_log_folder(self):
        d = ensure_log_dir()
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(d)))


    def show_about(self):
        QtWidgets.QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"""
        <b>{APP_NAME}</b><br>
        Version: {APP_VERSION}<br>
        Build date: {BUILD_DATE}<br><br>

        Python-based GUI for Optris IR cameras<br>
        University of Basel<br>
        by Paul Hiret<br><br>

        © 2026
        """,
        )

    # ---------------- display ----------------
    def _show_bundle(self, bundle: FrameBundle):
        temp_c, rad = self._bundle_view(bundle)
        img = temp_c if self.mode == "Temperature" else rad
        raw = bundle.raw_u16

        if self.scale_auto:
            vmin = float(np.nanmin(img))
            vmax = float(np.nanmax(img))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
                vmin, vmax = 0.0, 1.0
        else:
            vmin = float(self.scale_min)
            vmax = float(self.scale_max)

        label = "°C" if self.mode == "Temperature" else "Radiance"
        self.img_widget.set_image(img, raw, label=label, vmin=vmin, vmax=vmax, cmap=self.palette)

    def _update_measurements(self, bundle: FrameBundle):
        temp_c, rad = self._bundle_view(bundle)
        mean_t = float(np.nanmean(self._roi_slice(temp_c)))
        mean_r = float(np.nanmean(self._roi_slice(rad)))
        self.ed_mean_temp.setText(f"{mean_t:.2f}")
        self.ed_mean_rad.setText(f"{mean_r:.6e}")

    def refresh_display(self):
        bundle = self._current_bundle()
        if bundle is None:
            return
        self._update_measurements(bundle)
        self._show_bundle(bundle)

    # ---------------- camera connect ----------------
    def connect_camera(self):
        try:
            self.cam.connect()
            self.cam.set_temperature_range(*DEFAULT_T_RANGE)
            self._set_connected(True)
            self._log("Connected to camera")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Connect error", str(e))
            self._log(f"Connect error: {e}")

    def disconnect_camera(self):
        self.stop_live()
        try:
            self.cam.terminate()
        except Exception as e:
            self._log(f"Terminate error: {e}")
        self._set_connected(False)
        self._log("Disconnected")

    # ---------------- params ----------------
    def on_params_changed(self):
        self.cam.emissivity = float(self.spin_eps.value())
        self.cam.transmission = float(self.spin_tau.value())
        self.refresh_display()

    def on_mode_changed(self, txt: str):
        self.mode = txt
        self.refresh_display()

    def on_palette_changed(self, txt: str):
        # ✅ palette change now works because this is a Matplotlib colormap
        self.palette = txt
        self.img_widget.set_cmap(txt)
        self.refresh_display()

    def on_range_changed(self, txt: str):
        if not self.cam.connected:
            return
        try:
            if txt == "(150, 900)":
                self.cam.set_temperature_range(150, 900)
            elif txt == "(0, 250)":
                self.cam.set_temperature_range(0, 250)
            elif txt == "(-20, 100)":
                self.cam.set_temperature_range(-20, 100)
            self._log(f"Temperature range set: {txt}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Range error", str(e))
            self._log(f"Range error: {e}")

    # ---------------- acquisition ----------------
    def toggle_live(self):
        if self.live_enabled:
            self.stop_live()
        else:
            self.start_live()

    def start_live(self):
        if not self.cam.connected:
            self._log("Cannot start live: not connected")
            return
        self.live_enabled = True
        self.btn_live.setText("Stop Live")
        self.timer.start(int(1000 / DEFAULT_FPS))
        self._log("Live started")

    def stop_live(self):
        self.live_enabled = False
        self.btn_live.setText("Start Live")
        self.timer.stop()
        if self.recording:
            self.recording = False
            self.lamp_recording.setText("Not Recording")
            self.lamp_recording.setStyleSheet("background:#666666; color:white; font-weight:600; padding:4px; border-radius:6px;")
        self._log("Live stopped")

    def on_tick(self):
        try:
            bundle = self.cam.capture_one()
            self.last_bundle = bundle
        except Exception as e:
            self.stop_live()
            QtWidgets.QMessageBox.critical(self, "Acquisition error", str(e))
            self._log(f"Acquisition error: {e}")
            return

        if bundle.raw_u16 is not None:
            uniq = int(np.unique(bundle.raw_u16).size)
            if uniq <= 1:
                self._log(f"WARNING: RAW constant (unique={uniq})")

        if self.recording:
            self.frames.append(bundle)
            self.slider.setMaximum(max(0, len(self.frames) - 1))
            self.slider.setValue(len(self.frames) - 1)

        self._update_measurements(bundle)
        self._show_bundle(bundle)

    def capture(self):
        if not self.cam.connected:
            self._log("Capture ignored: not connected")
            return
        try:
            bundle = self.cam.capture_one()
            self.last_bundle = bundle
            self.frames = [bundle]
            self.current_index = 0
            self.slider.setMaximum(0)
            self.slider.setValue(0)
            self.refresh_display()
            self._log("Captured 1 frame")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Capture error", str(e))
            self._log(f"Capture error: {e}")

    def toggle_record(self):
        if not self.cam.connected:
            self._log("Record ignored: not connected")
            return
        self.recording = not self.recording
        if self.recording:
            self.frames = []
            self.slider.setValue(0)
            self.slider.setMaximum(0)
            self.lamp_recording.setText("Recording")
            self.lamp_recording.setStyleSheet("background:#ffaa00; color:black; font-weight:700; padding:4px; border-radius:6px;")
            self._log("Recording ON")
        else:
            self.lamp_recording.setText("Not Recording")
            self.lamp_recording.setStyleSheet("background:#666666; color:white; font-weight:600; padding:4px; border-radius:6px;")
            self._log(f"Recording OFF ({len(self.frames)} frames)")

    def on_slider(self, idx: int):
        self.current_index = idx
        self.refresh_display()

    # ---------------- emissivity tools ----------------
    def compute_emissivity_planck(self):
        bundle = self._current_bundle()
        if bundle is None:
            self._log("Planck emissivity: no data")
            return

        Tc = float(self.spin_tc.value())
        T = Tc + 273.15

        _, rad = self._bundle_view(bundle)
        measured_radiance = float(np.nanmean(self._roi_slice(rad)))
        bb = planck_integrated_radiance_75_14um(T)
        eps = measured_radiance / bb if bb > 0 else float("nan")
        self.ed_emi_planck.setText(f"{eps:.6f}")
        self._log(f"Planck emissivity: {eps:.6f}")

    def compute_emissivity_bb(self):
        bundle = self._current_bundle()
        if bundle is None:
            self._log("BB emissivity: no data")
            return
        bb = float(self.spin_bb_rad.value())
        if bb <= 0:
            self._log("BB emissivity: radiance must be > 0")
            return
        _, rad = self._bundle_view(bundle)
        measured_radiance = float(np.nanmean(self._roi_slice(rad)))
        eps = measured_radiance / bb
        self.ed_emi_bb.setText(f"{eps:.6f}")
        self._log(f"BB emissivity: {eps:.6f}")

    # ---------------- saving CSV ----------------
    def save_csv_raw(self):
        bundle = self._current_bundle()
        if bundle is None or bundle.raw_u16 is None:
            self._log("Save RAW CSV: no raw available")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save RAW uint16 CSV", "raw_u16.csv", "CSV (*.csv)")
        if not path:
            return
        np.savetxt(path, bundle.raw_u16.astype(np.uint16), delimiter=",", fmt="%d")
        self._log(f"Saved RAW CSV: {os.path.basename(path)}")

    def save_csv_temp_wo(self):
        bundle = self._current_bundle()
        if bundle is None:
            self._log("Save Temp woEmi CSV: no data")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Temp woEmi CSV", "temp_woEmi_C.csv", "CSV (*.csv)")
        if not path:
            return
        np.savetxt(path, bundle.temp_c_wo_emi, delimiter=",", fmt="%.6f")
        self._log(f"Saved Temp woEmi CSV: {os.path.basename(path)}")

    def save_csv_temp(self):
        bundle = self._current_bundle()
        if bundle is None:
            self._log("Save corrected CSV: no data")
            return
        temp_c, _ = self._bundle_view(bundle)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save corrected Temp CSV", "temp_corrected_C.csv", "CSV (*.csv)")
        if not path:
            return
        np.savetxt(path, temp_c, delimiter=",", fmt="%.6f")
        self._log(f"Saved corrected CSV: {os.path.basename(path)}")

    # ---------------- Save single TIFF ----------------
    def save_tiff_single(self):
        bundle = self._current_bundle()
        if bundle is None:
            self._log("Save TIFF: no data")
            return

        temp_c, rad = self._bundle_view(bundle)
        img = temp_c if self.mode == "Temperature" else rad

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save TIFF", "frame.tiff", "TIFF (*.tif *.tiff)")
        if not path:
            return

        # Prefer scientific float32 TIFF if tifffile is installed
        try:
            import tifffile
            tifffile.imwrite(path, img.astype(np.float32))
            self._log(f"Saved float TIFF: {os.path.basename(path)}")
            return
        except Exception:
            pass

        # Fallback: save a colored TIFF using the current colormap mapping
        vmin, vmax = (float(np.nanmin(img)), float(np.nanmax(img))) if self.scale_auto else (self.scale_min, self.scale_max)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            vmin, vmax = 0.0, 1.0

        norm = (img - vmin) / (vmax - vmin)
        u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)  # ok fallback
        cv2.imwrite(path, colored)
        self._log(f"Saved colored TIFF (fallback): {os.path.basename(path)}")

    # ---------------- HDF5 save/load ----------------
    def save_h5(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(self, "Nothing to save", "No recorded frames to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save dataset", "", "HDF5 (*.h5)")
        if not path:
            return
        try:
            with h5py.File(path, "w") as f:
                f.attrs["emissivity"] = float(self.cam.emissivity)
                f.attrs["transmission"] = float(self.cam.transmission)
                f.attrs["mode"] = self.mode
                f.attrs["palette"] = self.palette
                f.attrs["scale_auto"] = int(self.scale_auto)
                f.attrs["scale_min"] = float(self.scale_min)
                f.attrs["scale_max"] = float(self.scale_max)
                if self.roi is not None:
                    f.attrs["roi"] = np.array(self.roi, dtype=np.int32)

                tempwo = np.stack([b.temp_c_wo_emi for b in self.frames], axis=0)
                f.create_dataset("temp_c_wo_emi", data=tempwo, compression="gzip")

                if all(b.raw_u16 is not None for b in self.frames):
                    raw = np.stack([b.raw_u16 for b in self.frames], axis=0)
                    f.create_dataset("raw_u16", data=raw, compression="gzip")

            self._log(f"Saved {len(self.frames)} frames to {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save error", str(e))
            self._log(f"Save error: {e}")

    def open_h5(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open dataset", "", "HDF5 (*.h5)")
        if not path:
            return
        try:
            with h5py.File(path, "r") as f:
                tempwo = f["temp_c_wo_emi"][...]
                raw = f["raw_u16"][...] if "raw_u16" in f else None

                self.frames = []
                for i in range(tempwo.shape[0]):
                    r = raw[i] if raw is not None else None
                    self.frames.append(FrameBundle(raw_u16=r, temp_c_wo_emi=tempwo[i]))

                self.spin_eps.setValue(float(f.attrs.get("emissivity", 1.0)))
                self.spin_tau.setValue(float(f.attrs.get("transmission", 1.0)))

                m = f.attrs.get("mode", "Temperature")
                if isinstance(m, bytes):
                    m = m.decode("utf-8")
                self.combo_mode.setCurrentText(str(m))

                p = f.attrs.get("palette", "turbo")
                if isinstance(p, bytes):
                    p = p.decode("utf-8")
                if str(p) in [self.combo_palette.itemText(i) for i in range(self.combo_palette.count())]:
                    self.combo_palette.setCurrentText(str(p))

                self.scale_auto = bool(int(f.attrs.get("scale_auto", 1)))
                self.scale_min = float(f.attrs.get("scale_min", 0.0))
                self.scale_max = float(f.attrs.get("scale_max", 1.0))

                self.chk_auto.setChecked(self.scale_auto)
                self.spin_cmin.setValue(self.scale_min)
                self.spin_cmax.setValue(self.scale_max)
                self.on_scale_changed()

                roi = f.attrs.get("roi", None)
                if roi is not None:
                    roi = tuple(int(x) for x in np.array(roi).tolist())
                    self.roi = roi
                    self.lbl_roi.setText(f"ROI: x[{roi[0]}:{roi[2]}] y[{roi[1]}:{roi[3]}]")
                else:
                    self.roi = None
                    self.lbl_roi.setText("ROI: none")

            self.slider.setMaximum(max(0, len(self.frames) - 1))
            self.slider.setValue(0)
            self.current_index = 0
            self.last_bundle = self.frames[0] if self.frames else None
            self.refresh_display()
            self._log(f"Loaded {len(self.frames)} frames from {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open error", str(e))
            self._log(f"Open error: {e}")

    # ---------------- AVI export ----------------
    def export_avi(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "No frames to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export AVI", "", "AVI (*.avi)")
        if not path:
            return
        try:
            temp0, rad0 = self._bundle_view(self.frames[0])
            h, w = temp0.shape
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            vw = cv2.VideoWriter(path, fourcc, float(DEFAULT_FPS), (w, h))

            fixed = not self.scale_auto
            vmin_fixed = float(self.scale_min)
            vmax_fixed = float(self.scale_max)

            for b in self.frames:
                temp_c, rad = self._bundle_view(b)
                img = temp_c if self.mode == "Temperature" else rad

                if fixed:
                    vmin, vmax = vmin_fixed, vmax_fixed
                else:
                    vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))

                if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
                    vmin, vmax = 0.0, 1.0

                norm = (img - vmin) / (vmax - vmin)
                u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)

                col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
                vw.write(col)

            vw.release()
            self._log(f"Exported AVI: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export error", str(e))
            self._log(f"Export error: {e}")

    def closeEvent(self, event):
        try:
            self.stop_live()
            self.cam.terminate()
        except Exception:
            pass
        self._log("App closed")
        super().closeEvent(event)

    def apply_theme(self, theme: str):
        app = QtWidgets.QApplication.instance()
        if theme == "Dark":
            pal = QtGui.QPalette()
            pal.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
            pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.Base, QtGui.QColor(20, 20, 20))
            pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(35, 35, 35))
            pal.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 45))
            pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
            pal.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 120, 215))
            pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
            app.setPalette(pal)
        else:
            app.setPalette(app.style().standardPalette())

        self._log(f"Theme set: {theme}")


# ============================================================
# Splash + main
# ============================================================
def main():
    dll_path, cfg = _check_paths()
    cam = OptrisIR(dll_path, cfg)

    app = QtWidgets.QApplication(sys.argv)

    # Set global icon (taskbar + window)
    if ICON_PATH.exists():
        app.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))

    splash = None
    if SPLASH_PATH.exists():
        pix = QtGui.QPixmap(str(SPLASH_PATH))
        splash = QtWidgets.QSplashScreen(pix)
        splash.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        splash.show()
        app.processEvents()

    w = MainWindow(cam)
    w.resize(1500, 900)
    w.show()

    if splash is not None:
        splash.finish(w)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
