import sys
import math
import win32api
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QHBoxLayout,
                              QProgressBar, QStackedWidget)
from PyQt5.QtGui import QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QRectF

WIN_X = 0       # left edge of primary monitor
WIN_Y = 930     # top edge
WIN_W = 1140    # width
WIN_H = 100     # height

FONT_FAMILY = "Helvetica"
FONT_SIZE   = 16

LAYOUT_MARGIN   = 30   # left and right outer margin of the row
LAYOUT_SPACING  = 40   # gap between adjacent widgets
VERTICAL_PADDING = 10  # padding from top and bottom edges

FPS_EMA_ALPHA = 0.05

EPS_STEER = 0.005
EPS_ACCEL = 0.005
EPS_BRAKE = 0.005
EPS_SPEED = 0.5
EPS_FPS   = 0.3

ARC_SPAN = 160.0    # total degrees (±80° from 12-o'clock)

class SteeringArc(QWidget):
    def __init__(self, size: int, parent=None):
        super().__init__(parent)
        width = int(size * 1.8)
        self.setFixedSize(width, size)
        self._s   = size
        self._w   = width
        self._def = 0.0

    def set_steering(self, value: float):
        """value ∈ [0,1]: 0=full-left, 0.5=straight, 1=full-right."""
        nd = (value - 0.5) * ARC_SPAN
        if abs(nd - self._def) > 0.5:
            self._def = nd
            self.update()

    def _color(self) -> QColor:
        d = min(abs(self._def) / (ARC_SPAN / 2), 1.0)
        if d < 0.5:
            t = d * 2.0
            return QColor(int(t * 255), int(191 - t * 141), int(255 - t * 255))
        t = (d - 0.5) * 2.0
        return QColor(255, int(50 - t * 50), 0)

    def paintEvent(self, _):
        s  = self._s  # height
        w  = self._w  # width
        cx = w / 2.0  # center horizontally in the widget
        r  = s * 0.7
        pw = max(2, int(s * 0.09))
        cy = s - r * 0.174 - pw/2
        
        p  = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(cx - r, cy - r, r * 2, r * 2)

        # Background track
        p.setPen(QPen(QColor("#2a2a2a"), pw, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, int((90 + ARC_SPAN / 2) * 16), int(-ARC_SPAN * 16))

        # Centre tick
        p.setPen(QPen(QColor("#444444"), 2))
        p.drawLine(int(cx), int(cy - r + pw), int(cx), int(cy - r + pw * 2))

        # Swept arc
        if abs(self._def) > 0.5:
            p.setPen(QPen(self._color(), pw, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(rect, int(90 * 16), int(-self._def * 16))

        # Needle
        nr  = r * 0.70
        ang = math.radians(self._def)
        p.setPen(QPen(self._color() if abs(self._def) > 0.5 else QColor("white"),
                      3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(int(cx), int(cy),
                   int(cx + nr * math.sin(ang)), int(cy - nr * math.cos(ang)))

        # L / R labels
        p.setPen(QPen(QColor("white"), 1))
        p.setFont(QFont(FONT_FAMILY, 12, QFont.Bold))
        p.drawText(QRectF(2, cy - 8, 12, 16), Qt.AlignCenter, "L")
        p.drawText(QRectF(w - 14, cy - 8, 12, 16), Qt.AlignCenter, "R")
        p.end()

class VBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOrientation(Qt.Vertical)
        self.setRange(0, 100)
        self.setValue(0)
        self.setFixedWidth(18)
        self.setTextVisible(False)
        self._last = -1

    def set_value(self, v: float):
        iv = int(v * 100)
        if iv != self._last:
            self.setValue(iv)
            self._last = iv

_OV_NONE  = 0
_OV_PAUSE = 1
_OV_HUMAN = 2
_OV_SEG   = 3

class OverlayPanel(QStackedWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: black;")
        self.addWidget(self._lbl("",        "black",  14))
        self.addWidget(self._lbl("PAUSED!", "yellow", 22))
        self._human = self._lbl("", "red",   18); self.addWidget(self._human)
        self._seg   = self._lbl("", "white", 16); self.addWidget(self._seg)
        self.setCurrentIndex(_OV_NONE)

    @staticmethod
    def _lbl(text, color, size):
        w = QLabel(text)
        w.setAlignment(Qt.AlignCenter)
        w.setWordWrap(True)
        w.setStyleSheet(
            f"color:{color};background:black;"
            f"font-family:{FONT_FAMILY};font-size:{size}pt;font-weight:bold;")
        return w

    def show_pause(self):           self.setCurrentIndex(_OV_PAUSE)
    def show_human(self, t):        self._human.setText(t); self.setCurrentIndex(_OV_HUMAN)
    def show_segmentation(self, t): self._seg.setText(t);   self.setCurrentIndex(_OV_SEG)
    def hide_overlay(self):         self.setCurrentIndex(_OV_NONE)


class GUI:
    def __init__(self):
        self.run  = True
        
        if not QApplication.instance():
            QApplication.setAttribute(Qt.AA_DisableHighDpiScaling, True)
        
        self._app = QApplication.instance() or QApplication(sys.argv)

        self._win = QWidget()
        self._win.setWindowTitle("Telemetry")
        self._win.setStyleSheet("background: black;")
        self._win.setGeometry(WIN_X, WIN_Y, WIN_W, WIN_H)
        self._win.setFixedSize(WIN_W, WIN_H)
        self._win.closeEvent = lambda e: self._on_close(e)

        # Calculate usable height after accounting for vertical padding
        usable_height = WIN_H - (2 * VERTICAL_PADDING)
        arc_size = usable_height
        self._arc = SteeringArc(arc_size)

        ls = (f"color:white;background:black;"
              f"font-family:{FONT_FAMILY};font-size:{FONT_SIZE}pt;")

        self._steer_lbl = QLabel("Steering: 0.00");       self._steer_lbl.setStyleSheet(ls)
        self._accel_lbl = QLabel("Acceleration: 0.00");   self._accel_lbl.setStyleSheet(ls)
        self._brake_lbl = QLabel("Brake: 0.00");          self._brake_lbl.setStyleSheet(ls)
        self._speed_lbl = QLabel("Speed: 0.0");           self._speed_lbl.setStyleSheet(ls)
        self._fps_lbl   = QLabel("FPS: 0.0");             self._fps_lbl.setStyleSheet(ls)
        self._accel_bar = VBar()
        self._accel_bar.setFixedHeight(usable_height)
        self._brake_bar = VBar()
        self._brake_bar.setFixedHeight(usable_height)

        row = QHBoxLayout()
        row.setContentsMargins(LAYOUT_MARGIN, VERTICAL_PADDING, LAYOUT_MARGIN, VERTICAL_PADDING)
        row.setSpacing(LAYOUT_SPACING)
        row.addWidget(self._steer_lbl)
        row.addWidget(self._arc, alignment=Qt.AlignBottom)
        row.addWidget(self._accel_lbl)
        row.addWidget(self._accel_bar, alignment=Qt.AlignBottom)
        row.addWidget(self._brake_lbl)
        row.addWidget(self._brake_bar, alignment=Qt.AlignBottom)
        row.addWidget(self._speed_lbl)
        row.addWidget(self._fps_lbl)
        row.addStretch(1)

        self._telemetry = QWidget()
        self._telemetry.setLayout(row)
        self._telemetry.setStyleSheet("background: black;")

        self._overlay = OverlayPanel()

        self._stack = QStackedWidget()
        self._stack.addWidget(self._telemetry)
        self._stack.addWidget(self._overlay)
        self._stack.setCurrentIndex(0)

        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._stack)
        self._win.setLayout(root)
        self._win.show()

        self.human_intervening         = False
        self.segmentation_intervening  = False
        self.paused                    = False
        self.track_human_interventions = False

        self._last_steering     = None
        self._last_acceleration = None
        self._last_brake        = None
        self._last_speed        = None
        self._last_fps          = None
        self._ft_ema            = None

        self._EPS_STEER = EPS_STEER
        self._EPS_ACCEL = EPS_ACCEL
        self._EPS_BRAKE = EPS_BRAKE
        self._EPS_SPEED = EPS_SPEED
        self._EPS_FPS   = EPS_FPS
    

    def _on_close(self, e):
        self.run = False
        e.accept()

    def _show_overlay(self):
        self._stack.setCurrentIndex(1)

    def _hide_overlay(self):
        self._stack.setCurrentIndex(0)
        self._last_steering = self._last_acceleration = self._last_brake = None
        self._last_speed = self._last_fps = None

    def _smooth_fps(self, raw_fps: float) -> float:
        """EMA on frame-time then invert"""
        if raw_fps <= 0.0:
            return (1.0 / self._ft_ema) if self._ft_ema else 0.0
        ft = 1.0 / raw_fps
        self._ft_ema = ft if self._ft_ema is None else (
            FPS_EMA_ALPHA * ft + (1.0 - FPS_EMA_ALPHA) * self._ft_ema)
        return 1.0 / self._ft_ema
    

    def exit(self):
        self.run = False
        self._win.close()

    def show_pause_indicator(self):
        self._overlay.show_pause()
        self._show_overlay()

    def show_intervention_warning(self, action_text: str):
        if not self.track_human_interventions or self.paused:
            return
        self._overlay.show_human(f"HUMAN INTERVENTION DETECTED!\n{action_text}")
        self._show_overlay()

    def show_segmentation_intervention(self, action_type: str, action_details: str = ""):
        if self.paused:
            return
        msg = f"SEGMENTATION ASSISTANCE: {action_type}"
        if action_details:
            msg += f" by {action_details}"
        self._overlay.show_segmentation(msg)
        self._show_overlay()

    def restore_normal_display(self):
        self._overlay.hide_overlay()
        self._hide_overlay()

    def get_action_text(self, kw, ka, ks, kd) -> str:
        motion = ("Accelerating and braking" if kw and ks
                  else "Accelerating" if kw else "Braking" if ks else "")
        steer  = ("steering left and right" if ka and kd
                  else "steering left" if ka else "steering right" if kd else "")
        if motion and steer:  return f"{motion} and {steer}!"
        if motion:            return f"{motion}!"
        if steer:             return f"{steer.capitalize()}!"
        return "Unknown action!"

    def check_human_intervention(self) -> bool:
        if self.paused or not self.track_human_interventions:
            return False
        kw = bool(win32api.GetAsyncKeyState(0x57) & 0x8000)
        ka = bool(win32api.GetAsyncKeyState(0x41) & 0x8000)
        ks = bool(win32api.GetAsyncKeyState(0x53) & 0x8000)
        kd = bool(win32api.GetAsyncKeyState(0x44) & 0x8000)
        iv = kw or ka or ks or kd
        if iv:
            self.human_intervening = True
            self.show_intervention_warning(self.get_action_text(kw, ka, ks, kd))
        elif self.human_intervening:
            self.human_intervening = False
            if not self.segmentation_intervening:
                self._overlay.hide_overlay(); self._hide_overlay()
        return iv

    def check_segmentation_intervention(self, seg) -> bool:
        if self.paused:
            return False
        iv = False; itype = None; details = ""
        if seg:
            sa = seg.get("steer", "maintain")
            sp = seg.get("speed", "maintain")
            if sa != "maintain" or sp != "maintain":
                iv = True
                pri = seg.get("priority", 0)
                itype = ("Avoiding pedestrian" if pri == 3
                         else "Avoiding vehicle" if pri == 2
                         else "Avoiding road exit" if pri == 1
                         else "Assist active")
                parts = ([f"Steering {sa}"] if sa != "maintain" else [])
                if sp == "stop":   parts.append("Emergency braking")
                elif sp == "slow": parts.append("Slowing down")
                details = " and ".join(parts)
        if iv:
            if self.segmentation_intervening != itype:
                self.segmentation_intervening = itype
            self.show_segmentation_intervention(itype, details)
        elif self.segmentation_intervening:
            self.segmentation_intervening = False
            if not self.human_intervening:
                self._overlay.hide_overlay(); self._hide_overlay()
        return iv

    def set_pause_state(self, is_paused: bool):
        if is_paused == self.paused:
            return
        self.paused = is_paused
        if is_paused:
            self.show_pause_indicator()
        else:
            kw = bool(win32api.GetAsyncKeyState(0x57) & 0x8000)
            ka = bool(win32api.GetAsyncKeyState(0x41) & 0x8000)
            ks = bool(win32api.GetAsyncKeyState(0x53) & 0x8000)
            kd = bool(win32api.GetAsyncKeyState(0x44) & 0x8000)
            if self.human_intervening and self.track_human_interventions:
                self.show_intervention_warning(self.get_action_text(kw, ka, ks, kd))
            elif self.segmentation_intervening:
                self.show_segmentation_intervention(self.segmentation_intervening)
            else:
                self._overlay.hide_overlay(); self._hide_overlay()

    def update(self, steering: float, acceleration: float, brake: float,
               speed: float, fps: float,
               segmentation_action=None, is_paused: bool = False):
        
        if not self.run:
            return
        self.set_pause_state(is_paused)
        if not is_paused:
            hi = self.check_human_intervention()
            si = self.check_segmentation_intervention(segmentation_action)
            if not si and (not hi or not self.track_human_interventions):
                if self._last_steering is None or abs(steering - self._last_steering) > EPS_STEER:
                    self._steer_lbl.setText(f"Steering: {steering:.2f}")
                    self._last_steering = steering
                if self._last_acceleration is None or abs(acceleration - self._last_acceleration) > EPS_ACCEL:
                    self._accel_lbl.setText(f"Acceleration: {acceleration:.2f}")
                    self._accel_bar.set_value(acceleration)
                    self._last_acceleration = acceleration
                if self._last_brake is None or abs(brake - self._last_brake) > EPS_BRAKE:
                    self._brake_lbl.setText(f"Brake: {brake:.2f}")
                    self._brake_bar.set_value(brake)
                    self._last_brake = brake
                if self._last_speed is None or abs(speed - self._last_speed) > EPS_SPEED:
                    self._speed_lbl.setText(f"Speed: {speed:.1f}")
                    self._last_speed = speed
                smooth = self._smooth_fps(fps)
                if self._last_fps is None or abs(smooth - self._last_fps) > EPS_FPS:
                    self._fps_lbl.setText(f"FPS: {smooth:.1f}")
                    self._last_fps = smooth
                self._arc.set_steering(steering)
        self._app.processEvents()