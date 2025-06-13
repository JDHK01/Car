import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QFileDialog, QStyle, QSpinBox, QComboBox, QInputDialog
)
from PyQt5.QtCore import Qt
from video_logic import VideoLogic  # 修改为绝对导入

class VideoUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("通用视频处理")
        self.display_width = 640
        self.display_height = 480

        # QLabel控件
        self.label_single = QLabel()
        self.label_single.setFixedSize(self.display_width, self.display_height)
        self.label_single.setStyleSheet("background: #222;")
        self.label_single0 = QLabel()
        self.label_single0.setFixedSize(self.display_width, self.display_height)
        self.label_single0.setStyleSheet("background: #222;")

        # 按钮控件
        self.btn_open = QPushButton("打开视频")
        self.btn_camera = QPushButton("摄像头")
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_pause = QPushButton()
        self.btn_pause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.btn_pause.setEnabled(False)
        self.btn_close = QPushButton("关闭")
        self.btn_close.setEnabled(False)
        self.btn_snapshot = QPushButton("截图")
        self.btn_record = QPushButton("录制")
        self.btn_stoprec = QPushButton("停止录制")
        self.btn_exit = QPushButton("退出")

        # 帧率控件
        self.slider_fps = QSlider(Qt.Horizontal)
        self.slider_fps.setRange(1, 200)
        self.slider_fps.setValue(30)
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 200)
        self.spin_fps.setValue(30)
        self.slider_fps.valueChanged.connect(self.spin_fps.setValue)
        self.spin_fps.valueChanged.connect(self.slider_fps.setValue)

        # 图像选择
        self.combo_img = QComboBox()
        self.combo_img.addItems(["frame", "processed"])
        self.combo_img.setCurrentIndex(1)

        # 逻辑处理对象
        self.logic = VideoLogic(self)

        # 绑定信号
        self.btn_open.clicked.connect(self.logic.open_video)
        self.btn_camera.clicked.connect(self.logic.open_camera)
        self.btn_play.clicked.connect(self.logic.start_play)
        self.btn_pause.clicked.connect(self.logic.pause_play)
        self.btn_close.clicked.connect(self.logic.close_video)
        self.btn_snapshot.clicked.connect(self.logic.save_snapshot)
        self.btn_record.clicked.connect(self.logic.start_record)
        self.btn_stoprec.clicked.connect(self.logic.stop_record)
        self.btn_exit.clicked.connect(self.logic.exit_app)
        self.slider_fps.valueChanged.connect(self.logic.set_fps)
        self.combo_img.currentIndexChanged.connect(self.logic.update_image_display)

        # 布局
        hbox_btn0 = QHBoxLayout()
        hbox_btn0.addWidget(self.btn_open)
        hbox_btn0.addWidget(self.btn_camera)
        hbox_btn0.addWidget(self.btn_play)
        hbox_btn0.addWidget(self.btn_pause)
        hbox_btn0.addWidget(self.btn_close)
        hbox_btn0.addWidget(self.btn_exit)
        hbox_btn0.addWidget(QLabel("显示:"))
        hbox_btn0.addWidget(self.combo_img)
        hbox_btn1 = QHBoxLayout()
        hbox_btn1.addWidget(self.btn_snapshot)
        hbox_btn1.addWidget(self.btn_record)
        hbox_btn1.addWidget(self.btn_stoprec)
        hbox_fps = QHBoxLayout()
        hbox_fps.addWidget(QLabel("帧率:"))
        hbox_fps.addWidget(self.slider_fps)
        hbox_fps.addWidget(self.spin_fps)
        hbox_picture = QHBoxLayout()
        hbox_picture.addWidget(self.label_single0)
        hbox_picture.addWidget(self.label_single)
        vbox_ctrl = QVBoxLayout()
        vbox_ctrl.addLayout(hbox_btn0)
        vbox_ctrl.addLayout(hbox_btn1)
        vbox_ctrl.addLayout(hbox_fps)
        vbox_main = QVBoxLayout()
        vbox_main.addLayout(vbox_ctrl)
        vbox_main.addLayout(hbox_picture)
        self.setLayout(vbox_main)

    def keyPressEvent(self, event):
        self.logic.keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoUI()
    win.show()
    sys.exit(app.exec_())
