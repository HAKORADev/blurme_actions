import sys
import platform
import time
import os

import psutil
CPU_CORES = min(max(1, psutil.cpu_count(logical=False) or 1), 8)

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QObject, QMetaObject
from PyQt5.QtGui import QImage, QPainter

import numpy as np
import cv2
cv2.ocl.setUseOpenCL(True)
cv2.setUseOptimized(True)

import mss


class ConfigManager:
    def __init__(self):
        if getattr(sys, 'frozen', False):
            base_path = os.path.dirname(sys.executable)
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(base_path, "blur.conf")
        self.blur_mode = "colored"
        self.blurness  = 50
        self.opacity   = 255
        self.grayness  = 128
        self.load()

    def load(self):
        if not os.path.exists(self.config_path):
            self.save()
            return
        try:
            with open(self.config_path) as f:
                for line in f:
                    line = line.strip()
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key, value = key.strip(), value.strip()
                    if   key == "blur":     self.blur_mode = value
                    elif key == "blurness": self.blurness  = int(value)
                    elif key == "opacity":  self.opacity   = int(value)
                    elif key == "grayness": self.grayness  = int(value)
        except Exception:
            pass

    def save(self):
        try:
            with open(self.config_path, "w") as f:
                f.write(f"blur = {self.blur_mode}\n")
                f.write(f"blurness = {self.blurness}\n")
                f.write(f"opacity = {self.opacity}\n")
                f.write(f"grayness = {self.grayness}\n")
        except Exception:
            pass


class ScreenCaptureThread(QThread):
    image_ready = pyqtSignal(QImage)

    def __init__(self, blur_radius: int = 20, gray_mode: bool = False, gray_level: int = 128):
        super().__init__()
        self.blur_radius = blur_radius
        self.gray_mode   = gray_mode
        self.gray_level  = gray_level
        self.running     = True
        self.target_fps  = 30

    def _process_frame(self, frame_bgra):
        h, w = frame_bgra.shape[:2]

        if self.blur_radius > 0:
            sigma = self.blur_radius
            blurred = cv2.GaussianBlur(frame_bgra, (0, 0), sigmaX=sigma, sigmaY=sigma)
        else:
            blurred = frame_bgra.copy()

        if self.gray_mode and self.gray_level > 0:
            bgr = blurred[:, :, :3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            alpha = self.gray_level / 255.0
            blurred[:, :, :3] = cv2.addWeighted(bgr, 1.0 - alpha, gray3, alpha, 0)

        rgb = cv2.cvtColor(blurred, cv2.COLOR_BGRA2RGB)
        return np.ascontiguousarray(rgb)

    def run(self):
        frame_time = 1.0 / self.target_fps

        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                while self.running:
                    t0 = time.perf_counter()
                    try:
                        sct_img = sct.grab(monitor)
                        frame_bgra = np.asarray(sct_img)
                        processed  = self._process_frame(frame_bgra)
                        h, w       = processed.shape[:2]
                        qimg = QImage(processed.data, w, h, w * 3,
                                      QImage.Format_RGB888)

                        if self.running:
                            self.image_ready.emit(qimg.copy())

                    except Exception:
                        pass

                    elapsed    = time.perf_counter() - t0
                    sleep_time = max(0.001, frame_time - elapsed)
                    time.sleep(sleep_time)

        except Exception:
            pass

    def stop(self):
        self.running = False
        self.wait()


class BlurOverlay(QWidget):
    toggle_signal          = pyqtSignal()
    close_signal           = pyqtSignal()
    update_settings_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.config      = ConfigManager()
        self.blur_radius = int(self.config.blurness * 0.5)
        self.opacity     = self.config.opacity
        self.gray_mode   = self.config.blur_mode == "grayscale"
        self.gray_level  = self.config.grayness

        self.capture_thread    = None
        self.current_img       = None
        self.enabled           = False
        self.first_frame_ready = False

        self.toggle_signal.connect(self.toggle_effect,            Qt.QueuedConnection)
        self.close_signal.connect(self.safe_close,                Qt.QueuedConnection)
        self.update_settings_signal.connect(self.process_settings_update,
                                            Qt.QueuedConnection)
        self.setup_window()
        self.force_topmost()
        self.hide()

    def setup_window(self):
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.setWindowTitle("BlurMe")
        self.setWindowFlags(
            Qt.FramelessWindowHint       |
            Qt.WindowStaysOnTopHint      |
            Qt.WindowTransparentForInput |
            Qt.Tool                      |
            Qt.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_DeleteOnClose)

    def force_topmost(self):
        import subprocess
        wid = int(self.winId())
        script = f'tell application "System Events" to set frontmost of every process whose unix id is {os.getpid()} to true'
        subprocess.Popen(["osascript", "-e", script])

    @pyqtSlot()
    def toggle_effect(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.first_frame_ready = False
            self.start_capture()
        else:
            self.stop_capture()
            self.hide()
            self.current_img = None

    def start_capture(self):
        if self.capture_thread:
            self.capture_thread.stop()
        self.capture_thread = ScreenCaptureThread(
            self.blur_radius, self.gray_mode, self.gray_level
        )
        self.capture_thread.image_ready.connect(self.update_display)
        self.capture_thread.start()

    def stop_capture(self):
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None

    def update_display(self, qimg: QImage):
        self.current_img = qimg
        if self.enabled and not self.first_frame_ready:
            self.first_frame_ready = True
            self.show()
            self.raise_()
        self.update()

    def paintEvent(self, event):
        if not self.current_img:
            return
        painter = QPainter(self)
        painter.setOpacity(self.opacity / 255.0)
        painter.drawImage(self.rect(), self.current_img)
        painter.end()

    def process_settings_update(self, action: str):
        t = self.capture_thread

        if action == "f1":
            self.gray_mode = False
            self.config.blur_mode = "colored"
            if t: t.gray_mode = False
        elif action == "f2":
            self.gray_mode = True
            self.config.blur_mode = "grayscale"
            if t: t.gray_mode = True
        elif action == "f3":
            self.gray_level = max(0, self.gray_level - 10)
            self.config.grayness = self.gray_level
            if t: t.gray_level = self.gray_level
        elif action == "f4":
            self.gray_level = min(255, self.gray_level + 10)
            self.config.grayness = self.gray_level
            if t: t.gray_level = self.gray_level
        elif action == "minus":
            self.opacity = max(0, self.opacity - 4)
            self.config.opacity = self.opacity
        elif action == "plus":
            self.opacity = min(255, self.opacity + 4)
            self.config.opacity = self.opacity
        elif action == "slash":
            self.blur_radius = max(1, self.blur_radius - 1)
            self.config.blurness = int(self.blur_radius * 2)
            if t: t.blur_radius = self.blur_radius
        elif action == "asterisk":
            self.blur_radius = min(100, self.blur_radius + 1)
            self.config.blurness = int(self.blur_radius * 2)
            if t: t.blur_radius = self.blur_radius

        self.config.save()
        self.update()

    @pyqtSlot()
    def safe_close(self):
        self.stop_capture()
        self.close()
        os._exit(0)


class HotkeyManager(QObject):
    def __init__(self, overlay: BlurOverlay):
        super().__init__()
        self.overlay    = overlay
        self.pressed    = set()
        self.is_closing = False
        self._setup()

    def _setup(self):
        from pynput import keyboard

        char_actions = {'/': "slash", '*': "asterisk", '-': "minus", '+': "plus"}
        key_actions  = {
            keyboard.Key.f1: "f1", keyboard.Key.f2: "f2",
            keyboard.Key.f3: "f3", keyboard.Key.f4: "f4",
        }

        def on_press(key):
            if self.is_closing:
                return
            self.pressed.add(key)
            try:
                ctrl = any(k in self.pressed for k in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r))
                alt  = any(k in self.pressed for k in (keyboard.Key.alt_l,  keyboard.Key.alt_gr))
                if ctrl and alt:
                    char = getattr(key, 'char', None)
                    if char is None and getattr(key, 'vk', None) is not None:
                        char = {66: 'b', 67: 'c'}.get(key.vk)
                    if char:
                        c = char.lower()
                        if c == 'b':
                            self._emit("toggle")
                            return
                        if c == 'c':
                            self.is_closing = True
                            self._emit("close")
                            return
            except Exception:
                pass

            char = getattr(key, 'char', None)
            if char and char in char_actions:
                self._emit(char_actions[char])
            elif key in key_actions:
                self._emit(key_actions[key])

        def on_release(key):
            self.pressed.discard(key)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def _emit(self, action: str):
        if self.overlay is None:
            return
        try:
            if action == "toggle":
                QMetaObject.invokeMethod(self.overlay, "toggle_effect",
                                         Qt.QueuedConnection)
            elif action == "close":
                QMetaObject.invokeMethod(self.overlay, "safe_close",
                                         Qt.QueuedConnection)
            else:
                self.overlay.update_settings_signal.emit(action)
        except RuntimeError:
            pass


def main():
    app            = QApplication(sys.argv)
    overlay        = BlurOverlay()
    hotkey_manager = HotkeyManager(overlay)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
