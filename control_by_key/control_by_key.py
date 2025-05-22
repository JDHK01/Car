import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

# 导入你的麦克纳姆轮运动库
# sys.path.append('/home/pi/project_demo/lib')
# from McLumk_Wheel_Sports import *


class CarController(QWidget):
    '''
        坐标的参数描述：
            x:              左移为负数，右移为正数
            y:              前进为负数，后退为正数
            angle:          左转为负数，右转为正数
            move_speed:     移动速度，范围为0-255
            rotate_speed:   旋转速度，范围为0-255
    '''
    def __init__(self):
        super().__init__()
        self.x = 0
        self.y = 0
        self.angle = 0
        self.move_speed = 100
        self.rotate_speed = 100

        self.setWindowTitle("小车遥控器")
        # self.setGeometry(100, 100, 400, 300)
        self.init_ui()

    def init_ui(self):
        self.move_speed_label = QLabel(f"移动速度: {self.move_speed}")
        self.rotate_speed_label = QLabel(f"旋转速度: {self.rotate_speed}")

        layout = QVBoxLayout()
        layout.addWidget(self.move_speed_label)
        layout.addWidget(self.rotate_speed_label)

        # 滑块控制速度
        self.move_slider = QSlider(Qt.Horizontal)
        self.move_slider.setMinimum(0)
        self.move_slider.setMaximum(255)
        self.move_slider.setValue(self.move_speed)
        self.move_slider.valueChanged.connect(self.update_move_speed)
        layout.addWidget(QLabel("调整移动速度"))
        layout.addWidget(self.move_slider)

        self.rotate_slider = QSlider(Qt.Horizontal)
        self.rotate_slider.setMinimum(0)
        self.rotate_slider.setMaximum(255)
        self.rotate_slider.setValue(self.rotate_speed)
        self.rotate_slider.valueChanged.connect(self.update_rotate_speed)
        layout.addWidget(QLabel("调整旋转速度"))
        layout.addWidget(self.rotate_slider)

        # 实时显示x, y, angle
        self.status_label = QLabel(self.get_status_text())
        layout.addWidget(self.status_label)

        # 加入方向盘图标
        icon_layout = QHBoxLayout()
        icon_label = QLabel()
        # 你可以将steering_wheel.png放在同目录下，或修改路径
        pixmap = QPixmap("steering_wheel.png")
        pixmap = pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_layout.addStretch()
        icon_layout.addWidget(icon_label)
        icon_layout.addStretch()
        layout.addLayout(icon_layout)

        # 键盘控制提示
        layout.addWidget(QLabel("使用键盘控制：\nW：左移 | S：右移\nJ：前进 | K：后退\nA：左转 | D：右转"))

        self.setLayout(layout)

    def get_status_text(self):
        return f"x: {self.x}    y: {self.y}    angle: {self.angle}"

    def update_move_speed(self, value):
        self.move_speed = value
        self.move_speed_label.setText(f"移动速度: {value}")

    def update_rotate_speed(self, value):
        self.rotate_speed = value
        self.rotate_speed_label.setText(f"旋转速度: {value}")

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_J:
            self.y += self.move_speed
            # move_forward(self.move_speed)
        elif key == Qt.Key_K:
            self.y -= self.move_speed
            # move_backward(self.move_speed)
        elif key == Qt.Key_W:
            self.x -= self.move_speed
            # move_left(self.move_speed)
        elif key == Qt.Key_S:
            self.x += self.move_speed
            # move_right(self.move_speed)
        elif key == Qt.Key_A:
            self.angle -= self.rotate_speed
            # rotate_left(self.rotate_speed)
        elif key == Qt.Key_D:
            self.angle += self.rotate_speed
            # rotate_right(self.rotate_speed)

        self.status_label.setText(self.get_status_text())
        # QTimer.singleShot(300,print(f"x:{self.x};y:{self.y};angle:{self.angle}"))#, stop_robot)  # 自动停止

    def closeEvent(self, event):
        # stop_robot()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = CarController()
    controller.show()
    sys.exit(app.exec_())
