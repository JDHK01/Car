import cv2
import os
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QInputDialog, QApplication

def process_frame(frame):
    # 这里可以自定义处理逻辑
    processed = frame.copy()
    return frame, processed

class VideoLogic:
    def __init__(self, ui):
        self.ui = ui
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.playing = False
        self.locked_frame = None
        self.recording = False
        self.out = None
        self.img_idx = 0
        self.vid_idx = 0
        self._last_imgs = [None, None]
        os.makedirs('picture', exist_ok=True)
        os.makedirs('video', exist_ok=True)

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self.ui, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self._open_cap(path)

    def open_camera(self):
        cam_id, ok = QInputDialog.getItem(
            self.ui, "选择摄像头编号", "摄像头编号:", [str(i) for i in range(10)], 0, False
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
        self.ui.btn_close.setEnabled(True)
        self.ui.btn_play.setEnabled(True)
        self.ui.btn_pause.setEnabled(False)
        self.playing = False
        self.locked_frame = None
        self.show_frame()
        # 同理
        self.start_play()

    def show_frame(self, frame=None):
        if frame is None:
            if not self.cap:
                return
            ret, frame = self.cap.read()
            if not ret:
                return
            self.locked_frame = frame.copy()
        try:
            frame, processed = process_frame(frame)
        except Exception as e:
            print("process_frame error:", e)
            return
        self._last_imgs = [frame, processed]
        self.update_image_display()

    def update_image_display(self):
        idx = self.ui.combo_img.currentIndex()
        if self._last_imgs[idx] is not None:
            img = self._last_imgs[idx]
            self.set_image(self.ui.label_single, img)
            self.set_image(self.ui.label_single0, self._last_imgs[0])
        else:
            self.ui.label_single.clear()
            self.ui.label_single0.clear()

    def set_image(self, label, img):
        if img is None:
            label.clear()
            return
        img = cv2.resize(img, (self.ui.display_width, self.ui.display_height))
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def start_play(self):
        if not self.cap:
            return
        self.playing = True
        self.ui.btn_play.setEnabled(False)
        self.ui.btn_pause.setEnabled(True)
        self.timer.start(int(1000 / max(1, self.ui.slider_fps.value())))

    def pause_play(self):
        self.playing = False
        self.ui.btn_play.setEnabled(True)
        self.ui.btn_pause.setEnabled(False)
        self.timer.stop()

    def close_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.ui.btn_close.setEnabled(False)
        self.ui.btn_play.setEnabled(False)
        self.ui.btn_pause.setEnabled(False)
        self.clear_images()
        self.stop_record()

    def set_fps(self):
        if self.playing:
            self.timer.setInterval(int(1000 / max(1, self.ui.slider_fps.value())))

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
        self.ui.label_single.clear()
        self.ui.label_single0.clear()

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
        fps = self.ui.slider_fps.value()
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
