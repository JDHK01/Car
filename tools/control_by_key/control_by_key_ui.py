# ui.py

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer

# 导入功能函数
from control_by_key_logic import (
    move_forward, move_backward, move_left, move_right,
    rotate_left, rotate_right, stop_robot
)


class CarController(QWidget):
    def __init__(self):
        super().__init__()
        self.x = 0
        self.y = 0
        self.angle = 0
        self.move_speed = 100
        self.rotate_speed = 100

        self.setWindowTitle("小车遥控器")
        self.init_ui()

    def init_ui(self):
        self.move_speed_label = QLabel("移动速度:")
        self.move_speed_slider = QSlider(Qt.Horizontal)
        self.move_speed_slider.setRange(0, 255)
        self.move_speed_slider.setValue(self.move_speed)
        self.move_speed_box = QSpinBox()
        self.move_speed_box.setRange(0, 255)
        self.move_speed_box.setValue(self.move_speed)
        self.move_speed_slider.valueChanged.connect(self.move_speed_box.setValue)
        self.move_speed_box.valueChanged.connect(self.move_speed_slider.setValue)

        self.rotate_speed_label = QLabel("旋转速度:")
        self.rotate_speed_slider = QSlider(Qt.Horizontal)
        self.rotate_speed_slider.setRange(0, 255)
        self.rotate_speed_slider.setValue(self.rotate_speed)
        self.rotate_speed_box = QSpinBox()
        self.rotate_speed_box.setRange(0, 255)
        self.rotate_speed_box.setValue(self.rotate_speed)
        self.rotate_speed_slider.valueChanged.connect(self.rotate_speed_box.setValue)
        self.rotate_speed_box.valueChanged.connect(self.rotate_speed_slider.setValue)

        self.status_label = QLabel(self.get_status_text())

        layout1 = QHBoxLayout()
        layout1.addWidget(self.move_speed_label)
        layout1.addWidget(self.move_speed_slider)
        layout1.addWidget(self.move_speed_box)

        layout2 = QHBoxLayout()
        layout2.addWidget(self.rotate_speed_label)
        layout2.addWidget(self.rotate_speed_slider)
        layout2.addWidget(self.rotate_speed_box)

        v_layout = QVBoxLayout()
        v_layout.addLayout(layout1)
        v_layout.addLayout(layout2)
        v_layout.addWidget(self.status_label)
        v_layout.addWidget(QLabel("使用键盘控制：\nW：左移 | S：右移\nJ：前进 | K：后退\nA：左转 | D：右转"))

        self.setLayout(v_layout)

    def get_status_text(self):
        return f"x: {self.x}    y: {self.y}    angle: {self.angle}"

    def keyPressEvent(self, event):
        key = event.key()
        self.move_speed = self.move_speed_box.value()
        self.rotate_speed = self.rotate_speed_box.value()

        if key == Qt.Key_J:
            self.y += self.move_speed
            move_forward(self.move_speed)
        elif key == Qt.Key_K:
            self.y -= self.move_speed
            move_backward(self.move_speed)
        elif key == Qt.Key_W:
            self.x -= self.move_speed
            move_left(self.move_speed)
        elif key == Qt.Key_S:
            self.x += self.move_speed
            move_right(self.move_speed)
        elif key == Qt.Key_A:
            self.angle -= self.rotate_speed
            rotate_left(self.rotate_speed)
        elif key == Qt.Key_D:
            self.angle += self.rotate_speed
            rotate_right(self.rotate_speed)

        self.status_label.setText(self.get_status_text())
        QTimer.singleShot(300, lambda: print(f"x:{self.x}; y:{self.y}; angle:{self.angle}"))

    def closeEvent(self, event):
        stop_robot()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = CarController()
    controller.show()
    sys.exit(app.exec_())