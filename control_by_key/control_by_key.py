import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer


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
        self.init_ui()

    def init_ui(self):
        # ----------------------QLabel+QSlider+QSpinBox控件---------------------------------
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

        # ----------------QLabel控件-------------------------------
        self.status_label = QLabel(self.get_status_text())

        # ----------------------------布局------------------------
        QHBox_move_speed = QHBoxLayout()
        QHBox_move_speed.addWidget(self.move_speed_label)
        QHBox_move_speed.addWidget(self.move_speed_slider)
        QHBox_move_speed.addWidget(self.move_speed_box)

        QHBox_rotate_speed = QHBoxLayout()
        QHBox_rotate_speed.addWidget(self.rotate_speed_label)
        QHBox_rotate_speed.addWidget(self.rotate_speed_slider)
        QHBox_rotate_speed.addWidget(self.rotate_speed_box)

        QVBox = QVBoxLayout()
        QVBox.addLayout(QHBox_move_speed)
        QVBox.addLayout(QHBox_rotate_speed)
        QVBox.addWidget(self.status_label)
        QVBox.addWidget(QLabel("使用键盘控制：\nW：左移 | S：右移\nJ：前进 | K：后退\nA：左转 | D：右转"))

        self.setLayout(QVBox)

    def get_status_text(self):
        return f"x: {self.x}    y: {self.y}    angle: {self.angle}"

    def keyPressEvent(self, event):
        key = event.key()
        self.move_speed = self.move_speed_box.value()
        self.rotate_speed = self.rotate_speed_box.value()

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

        # 模拟定时停止或输出状态
        QTimer.singleShot(300, lambda: print(f"x:{self.x}; y:{self.y}; angle:{self.angle}"))
        # QTimer.singleShot(300, stop_robot)

    def closeEvent(self, event):
        # stop_robot()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = CarController()
    controller.show()
    sys.exit(app.exec_())
