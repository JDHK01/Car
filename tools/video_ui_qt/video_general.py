import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QFileDialog, QStyle, QSpinBox, QComboBox, QInputDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# -----------------保留的函数处理接口--------------------
def process_frame(frame):
    processed = frame.copy()
    return processed

class VideoUI(QWidget):
    def __init__(self):
        super().__init__()
        # -----------------------------ui名称------------------------
        self.setWindowTitle("通用视频处理")
        '''
        功能：
            cap  :视频捕获对象            cv2.VideoCapture()
            out  :视频输出对象            cv2.VideoWriter()
            timer:定时器，控制视频播放速度  QTimer()
            
        标志：
            playing：是否正在播放
            recording：是否正在录制
            
            img_idx：图片标识，避免覆盖
            vid_idx：视频标识，避免覆盖
            
            locked_frame:锁帧。对暂停案件的补充
            
        参数：
            display_width：显示区域宽度
            display_height：显示区域高度
            
        '''
        self.cap = None
        self.timer = QTimer();self.timer.timeout.connect(self.next_frame)
        self.playing = False
        self.locked_frame = None
        self.recording = False
        self.out = None
        self.img_idx = 0
        self.vid_idx = 0
        self.display_width = 640
        self.display_height = 480

        # -----------------------------ui控件------------------------
        '''
            QLabel控件：     label_single   显示区域设置
                            label_single0  显示原图设置
            QPushButton控件：btn_open      打开视频文件
                            btn_camera    打开摄像头
                            btn_play      播放
                            btn_pause     暂停
                            btn_close     关闭
                            btn_snapshot  截图
                            btn_record    录制
                            btn_stoprec   停止录制
                            btn_exit      退出
                            btn_play是标准播放控件，btn_pause是标准暂停控件
            QSlider控件+QSpinBox()：    
                            slider_fps    控制视频帧率    范围：1-200；默认值：30
                            spin_fps      控制视频帧率    范围：1-200；默认值：30
                            两者相互绑定
            QComboBox控件：  combo_img     选择显示图片
                            具体请自定义
                            
            QHBoxLayout控件：hbox_btn0     视频播放控件系列
                            hbox_btn1     录制截图控件系列
                            hbox_fps      帧率调节控件系列
            QVBoxLayout控件：vbox          整体布局
            
        '''
        # -----------------------QLabel控件------------------------------------
        self.label_single = QLabel()
        self.label_single.setFixedSize(self.display_width, self.display_height)
        self.label_single.setStyleSheet("background: #222;")

        self.label_single0 = QLabel()
        self.label_single0.setFixedSize(self.display_width, self.display_height)
        self.label_single0.setStyleSheet("background: #222;")
        # ----------------------QButton控件---------------------------------
        self.btn_open = QPushButton("打开视频")
        self.btn_open.clicked.connect(self.open_video)

        self.btn_camera = QPushButton("摄像头")
        self.btn_camera.clicked.connect(self.open_camera)

        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))# 直接使用媒体播放控件的标准图标
        self.btn_play.clicked.connect(self.start_play)

        self.btn_pause = QPushButton()
        self.btn_pause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))# 直接使用媒体暂停控件的标准图标
        self.btn_pause.setEnabled(False)    # 初始化是关闭的
        self.btn_pause.clicked.connect(self.pause_play)

        self.btn_close = QPushButton("关闭")
        self.btn_close.setEnabled(False)    # 初始化是关闭的
        self.btn_close.clicked.connect(self.close_video)

        self.btn_snapshot = QPushButton("截图")
        self.btn_snapshot.clicked.connect(self.save_snapshot)

        self.btn_record = QPushButton("录制")
        self.btn_record.clicked.connect(self.start_record)

        self.btn_stoprec = QPushButton("停止录制")
        self.btn_stoprec.clicked.connect(self.stop_record)


        self.btn_exit = QPushButton("退出")
        self.btn_exit.clicked.connect(self.exit_app)

        # ----------------------QSlider控件+QSpinBox控件---------------------------------
        self.slider_fps = QSlider(Qt.Horizontal)
        self.slider_fps.setRange(1, 200)
        self.slider_fps.setValue(30)
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 200)
        self.spin_fps.setValue(30)
        self.slider_fps.valueChanged.connect(self.spin_fps.setValue)
        self.spin_fps.valueChanged.connect(self.slider_fps.setValue)
        self.slider_fps.valueChanged.connect(self.set_fps)


        # -------------------QComboBox控件---------------------------------------
        self.combo_img = QComboBox()
        self.combo_img.addItems(["frame","processed"])
        self.combo_img.currentIndexChanged.connect(self.update_image_display)

        # ---------------------布局-----------------------------------
        hbox_btn0 = QHBoxLayout()
        hbox_btn0.setSpacing(2)
        hbox_btn0.setContentsMargins(2, 2, 2, 2)
        # 视频播放控件系列
        hbox_btn0.addWidget(self.btn_open)
        hbox_btn0.addWidget(self.btn_camera)
        hbox_btn0.addWidget(self.btn_play)
        hbox_btn0.addWidget(self.btn_pause)
        hbox_btn0.addWidget(self.btn_close)
        # hbox_btn0.addWidget(self.btn_snapshot)
        # hbox_btn0.addWidget(self.btn_record)
        # hbox_btn0.addWidget(self.btn_stoprec)
        hbox_btn0.addWidget(self.btn_exit)
        hbox_btn0.addWidget(QLabel("显示:"))
        hbox_btn0.addWidget(self.combo_img)
        # 录制截图控件系列
        hbox_btn1 = QHBoxLayout()
        hbox_btn1.setSpacing(2)
        hbox_btn1.setContentsMargins(2, 2, 2, 2)
        hbox_btn1.addWidget(self.btn_snapshot)
        hbox_btn1.addWidget(self.btn_record)
        hbox_btn1.addWidget(self.btn_stoprec)
        # 帧率调节控件系列
        hbox_fps = QHBoxLayout()
        hbox_fps.setSpacing(2)
        hbox_fps.setContentsMargins(2, 2, 2, 2)
        hbox_fps.addWidget(QLabel("帧率:"))
        hbox_fps.addWidget(self.slider_fps)
        hbox_fps.addWidget(self.spin_fps)
        # 画布
        hbox_picture = QHBoxLayout()
        hbox_picture.setSpacing(2)
        hbox_picture.setContentsMargins(2, 2, 2, 2)
        hbox_picture.addWidget(self.label_single0)
        hbox_picture.addWidget(self.label_single)
        # 三个水平布局融合
        vbox_ctrl = QVBoxLayout()
        vbox_ctrl.setSpacing(2)
        vbox_ctrl.setContentsMargins(4, 4, 4, 4)
        vbox_ctrl.addLayout(hbox_btn0)
        vbox_ctrl.addLayout(hbox_btn1)
        vbox_ctrl.addLayout(hbox_fps)
        # 再加入画布
        vbox_main = QVBoxLayout()
        vbox_main.setSpacing(6)
        vbox_main.setContentsMargins(6, 6, 6, 6)
        vbox_main.addLayout(vbox_ctrl)
        vbox_main.addLayout(hbox_picture)
        self.setLayout(vbox_main)
    '''
        按键说明：
            播放和暂停是二选一的关系
            关闭按键和cap的状态绑定。默认关闭，有cap对象后打开
            
        打开视频文件 = 选择视频源 + cap对象播放视频
            选择视频源：
                open_video()：借助 QFileDialog.getOpenFileName() 方法
                open_camera()：借助 QInputDialog.getItem() 方法
            cap对象播放视频：
                一系列流程
    '''
    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self._open_cap(path)

    def open_camera(self):
        cam_id, ok = QInputDialog.getItem(
            self, "选择摄像头编号", "摄像头编号:", [str(i) for i in range(10)], 0, False
        )
        if ok:
            self._open_cap(int(cam_id))

    def _open_cap(self, src):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.cap = None
            print("无法打开视频")
            return
        # 准备修改为打开视频自动播放:修改按键状态 + 自动按键
        self.btn_close.setEnabled(True)# 开启关闭按键
        self.btn_play.setEnabled(True)# 开启播放按键
        self.btn_pause.setEnabled(False)# 关闭暂停按键
        self.playing = False# 设置播放状态
        self.locked_frame = None
        self.show_frame()

        self.start_play()  # 自动播放

    def show_frame(self, frame=None):
        if frame is None:
            if not self.cap:
                return
            ret, frame = self.cap.read()
            if not ret:
                return
            self.locked_frame = frame.copy()

        try:
            # --------------------------图像处理----------------------
            processed = process_frame(frame)
        except Exception as e:
            print("process_frame error:", e)
            return
        # -----------------临时存储图像处理结果---------------------------
        self._last_imgs = [frame, processed]# 存储图像处理结果
        self.update_image_display()

    def update_image_display(self):
        idx = self.combo_img.currentIndex()
        # hasattr() 函数用于判断对象是否包含对应的属性
        # 实例有_last_imgs属性并且数值有效
        if hasattr(self, "_last_imgs") and self._last_imgs[idx] is not None:
            img = self._last_imgs[idx]
            # is_mask = idx != 0
            self.set_image(self.label_single, img)#is_mask=is_mask)
            self.set_image(self.label_single0, self._last_imgs[0])
        else:
            self.label_single.clear()
            self.label_single0.clear()

    def set_image(self, label, img): #is_mask=False):
        if img is None:
            label.clear()
            return
        # if is_mask:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, (self.display_width, self.display_height))
        h, w, ch = img.shape
        bytes_per_line = ch * w
        # Qt显示图片的核心
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qimg))

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
        self.stop_record()

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
        if self.recording and self.out is not None:
            self.out.write(frame)


    def clear_images(self):
        self.label_single.clear()
        self.label_single0.clear()

    def save_snapshot(self):
        if self.locked_frame is not None:
            fname = f'picture/traffic{self.img_idx}.png'
            cv2.imwrite(fname, self.locked_frame)
            print(f'保存图片 {fname}')
            self.img_idx += 1

    def start_record(self):
        if self.locked_frame is None or self.recording:
            print("未获取到画面或已在录制中")
            return
        h, w = self.locked_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = self.slider_fps.value()
        fname = f'video/traffic{self.vid_idx}.avi'
        self.out = cv2.VideoWriter(fname, fourcc, fps, (w, h))
        self.recording = True
        print(f"开始录制视频 {fname}")

    def stop_record(self):
        if self.recording and self.out is not None:
            self.recording = False
            self.out.release()
            print(f"停止录制 traffic{self.vid_idx}.avi")
            self.vid_idx += 1
            self.out = None

    def exit_app(self):
        self.close_video()
        QApplication.quit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            self.save_snapshot()
        elif event.key() == Qt.Key_R:
            self.start_record()
        elif event.key() == Qt.Key_E:
            self.stop_record()
        elif event.key() == Qt.Key_Q:
            self.exit_app()
        else:
            # 其他没有定义的按照父类的方法处理
            super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoUI()
    win.show()
    sys.exit(app.exec_())

