import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QFileDialog, QStyle, QSpinBox, QComboBox, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from lib.imgproc import process_frame

class RangeSlider(QWidget):
    """自定义双向滑块，带数值显示"""
    def __init__(self, min_val, max_val, init_low, init_high, label):
        super().__init__()
        self.label = QLabel(label)
        self.low = QSlider(Qt.Horizontal)
        self.high = QSlider(Qt.Horizontal)
        self.low.setMinimum(min_val)
        self.low.setMaximum(max_val)
        self.high.setMinimum(min_val)
        self.high.setMaximum(max_val)
        self.low.setValue(init_low)
        self.high.setValue(init_high)
        self.low_val = QSpinBox()
        self.high_val = QSpinBox()
        self.low_val.setRange(min_val, max_val)
        self.high_val.setRange(min_val, max_val)
        self.low_val.setValue(init_low)
        self.high_val.setValue(init_high)
        self.low.valueChanged.connect(self._low_changed)
        self.high.valueChanged.connect(self._high_changed)
        self.low_val.valueChanged.connect(self.low.setValue)
        self.high_val.valueChanged.connect(self.high.setValue)
        hbox = QHBoxLayout()
        hbox.setSpacing(4)
        hbox.setContentsMargins(2, 2, 2, 2)
        hbox.addWidget(self.label)
        hbox.addWidget(self.low)
        hbox.addWidget(self.low_val)
        hbox.addWidget(QLabel('~'))
        hbox.addWidget(self.high)
        hbox.addWidget(self.high_val)
        self.setLayout(hbox)
    def _low_changed(self, v):
        if v > self.high.value():
            self.low.setValue(self.high.value())
            return
        self.low_val.setValue(v)
    def _high_changed(self, v):
        if v < self.low.value():
            self.high.setValue(self.low.value())
            return
        self.high_val.setValue(v)
    def get(self):
        return [self.low.value(), self.high.value()]
    def set(self, low, high):
        self.low.setValue(low)
        self.high.setValue(high)

class VideoUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频HSV调节")
        self.cap = None
        self.timer = QTimer()
        self.playing = False
        self.locked_frame = None
        # 显示尺寸
        self.display_width = 800
        self.display_height = 600

        # 只保留一个图片显示区域（放大尺寸）
        self.label_single = QLabel()
        self.label_single.setFixedSize(self.display_width, self.display_height)
        self.label_single.setStyleSheet("background: #222;")

        # 控件
        self.btn_open = QPushButton("打开视频")
        self.btn_camera = QPushButton("摄像头")
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_pause = QPushButton()
        self.btn_pause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.btn_pause.setEnabled(False)
        self.btn_close = QPushButton("关闭")
        self.btn_close.setEnabled(False)

        self.slider_lowest = QSlider(Qt.Horizontal)
        self.slider_lowest.setRange(0, 100)
        self.slider_lowest.setValue(30)
        self.spin_lowest = QSpinBox()
        self.spin_lowest.setRange(0, 100)
        self.spin_lowest.setValue(30)
        self.slider_lowest.valueChanged.connect(self.spin_lowest.setValue)
        self.spin_lowest.valueChanged.connect(self.slider_lowest.setValue)

        self.slider_fps = QSlider(Qt.Horizontal)
        self.slider_fps.setRange(1, 200)
        self.slider_fps.setValue(30)
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 200)
        self.spin_fps.setValue(30)
        self.slider_fps.valueChanged.connect(self.spin_fps.setValue)
        self.spin_fps.valueChanged.connect(self.slider_fps.setValue)

        # HSV滑块
        self.red1_h = RangeSlider(0, 180, 0, 15, "red1 H")
        self.red1_s = RangeSlider(0, 255, 80, 255, "red1 S")
        self.red1_v = RangeSlider(0, 255, 80, 255, "red1 V")
        self.red2_h = RangeSlider(0, 180, 165, 180, "red2 H")
        self.red2_s = RangeSlider(0, 255, 80, 255, "red2 S")
        self.red2_v = RangeSlider(0, 255, 80, 255, "red2 V")

        # 图片选择下拉框
        self.combo_img = QComboBox()
        self.combo_img.addItems(["处理后", "原始掩膜", "腐蚀", "膨胀"])
        self.combo_img.currentIndexChanged.connect(self.update_image_display)

        # 布局
        hbox_btn = QHBoxLayout()
        hbox_btn.setSpacing(6)
        hbox_btn.setContentsMargins(2, 2, 2, 2)
        hbox_btn.addWidget(self.btn_open)
        hbox_btn.addWidget(self.btn_camera)
        hbox_btn.addWidget(self.btn_play)
        hbox_btn.addWidget(self.btn_pause)
        hbox_btn.addWidget(self.btn_close)
        hbox_btn.addWidget(QLabel("显示:"))
        hbox_btn.addWidget(self.combo_img)

        hbox_lowest = QHBoxLayout()
        hbox_lowest.setSpacing(4)
        hbox_lowest.setContentsMargins(2, 2, 2, 2)
        hbox_lowest.addWidget(QLabel("lowest:"))
        hbox_lowest.addWidget(self.slider_lowest)
        hbox_lowest.addWidget(self.spin_lowest)
        hbox_fps = QHBoxLayout()
        hbox_fps.setSpacing(4)
        hbox_fps.setContentsMargins(2, 2, 2, 2)
        hbox_fps.addWidget(QLabel("帧率:"))
        hbox_fps.addWidget(self.slider_fps)
        hbox_fps.addWidget(self.spin_fps)

        vbox_ctrl = QVBoxLayout()
        vbox_ctrl.setSpacing(4)
        vbox_ctrl.setContentsMargins(4, 4, 4, 4)
        vbox_ctrl.addLayout(hbox_btn)
        vbox_ctrl.addLayout(hbox_fps)
        vbox_ctrl.addLayout(hbox_lowest)
        vbox_ctrl.addWidget(self.red1_h)
        vbox_ctrl.addWidget(self.red2_h)
        vbox_ctrl.addWidget(self.red1_s)
        vbox_ctrl.addWidget(self.red2_s)
        vbox_ctrl.addWidget(self.red1_v)
        vbox_ctrl.addWidget(self.red2_v)

        vbox_main = QVBoxLayout()
        vbox_main.setSpacing(6)
        vbox_main.setContentsMargins(6, 6, 6, 6)
        vbox_main.addLayout(vbox_ctrl)
        vbox_main.addWidget(self.label_single)
        self.setLayout(vbox_main)

        # 信号
        self.btn_open.clicked.connect(self.open_video)
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_play.clicked.connect(self.start_play)
        self.btn_pause.clicked.connect(self.pause_play)
        self.btn_close.clicked.connect(self.close_video)
        self.slider_lowest.valueChanged.connect(self.refresh_frame)
        self.slider_fps.valueChanged.connect(self.set_fps)
        for s in [self.red1_h, self.red2_h, self.red1_s, self.red2_s, self.red1_v, self.red2_v]:
            s.low.valueChanged.connect(self.refresh_frame)
            s.high.valueChanged.connect(self.refresh_frame)
        self.timer.timeout.connect(self.next_frame)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self._open_cap(path)

    def open_camera(self):
        cam_id, ok = QInputDialog.getItem(
            self, "选择摄像头编号", "摄像头编号:", [str(i) for i in range(5)], 0, False
        )
        if ok:
            self._open_cap(int(cam_id))
    def _open_cap(self, src):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.cap = None
            return
        self.btn_close.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.playing = False
        self.locked_frame = None
        self.show_frame()

    def start_play(self):
        if not self.cap:
            return
        self.playing = True
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.timer.start(int(1000 / max(1, self.slider_fps.value())))

    def pause_play(self):
        self.playing = False
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.timer.stop()

    def close_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_close.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.clear_images()

    def set_fps(self):
        if self.playing:
            self.timer.setInterval(int(1000 / max(1, self.slider_fps.value())))

    def next_frame(self):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        self.locked_frame = frame.copy()
        self.show_frame(frame)

    def refresh_frame(self):
        # 暂停时，调节参数只对locked_frame处理
        if not self.cap:
            return
        if self.playing:
            return
        frame = self.locked_frame
        if frame is not None:
            self.show_frame(frame)

    def show_frame(self, frame=None):
        if frame is None:
            if not self.cap:
                return
            # 修正：去除未定义的pos变量，直接读取当前帧
            ret, frame = self.cap.read()
            if not ret:
                return
            self.locked_frame = frame.copy()
        temp = [
            [self.red1_h.get()[0], self.red1_s.get()[0], self.red1_v.get()[0]],
            [self.red1_h.get()[1], self.red1_s.get()[1], self.red1_v.get()[1]],
            [self.red2_h.get()[0], self.red2_s.get()[0], self.red2_v.get()[0]],
            [self.red2_h.get()[1], self.red2_s.get()[1], self.red2_v.get()[1]],
        ]
        try:
            draw, mask, mask8, mask9 = process_frame(
                frame,
                temp,
                lowest=self.slider_lowest.value()
            )
        except Exception as e:
            print("process_frame error:", e)
            return
        self._last_imgs = [draw, mask, mask8, mask9]
        self.update_image_display()

    def update_image_display(self):
        idx = self.combo_img.currentIndex()
        if hasattr(self, "_last_imgs") and self._last_imgs[idx] is not None:
            img = self._last_imgs[idx]
            is_mask = idx != 0
            self.set_image(self.label_single, img, is_mask=is_mask)
        else:
            self.label_single.clear()

    def set_image(self, label, img, is_mask=False):
        if img is None:
            label.clear()
            return
        if is_mask:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (self.display_width, self.display_height))
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def clear_images(self):
        self.label_single.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoUI()
    win.show()
    sys.exit(app.exec_())
