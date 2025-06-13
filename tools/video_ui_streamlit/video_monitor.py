import streamlit as st
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
import time

# 图像处理函数（可自定义）
def process_frame(frame):
    processed = frame.copy()
    return processed
    # return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 设置页面布局
st.set_page_config(page_title="通用视频处理", layout="wide")

st.title("📹 通用视频处理（Streamlit 版）")

# 选择视频源
source_type = st.sidebar.radio("选择视频源", ["本地视频", "摄像头"])

video_file = None
camera_index = 0
if source_type == "本地视频":
    video_file = st.sidebar.file_uploader("上传视频文件", type=["mp4", "avi", "mov"])
else:
    camera_index = st.sidebar.number_input("摄像头编号", min_value=0, max_value=10, value=0)

fps = st.sidebar.slider("帧率（播放速度）", 1, 60, 30)
show_original = st.sidebar.checkbox("显示原始视频", value=True)

start = st.button("开始播放")

if start:
    if source_type == "本地视频" and video_file is not None:
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(camera_index)

    stframe1 = st.empty()
    stframe2 = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame)

        # 显示图像（原始）
        if show_original:
            stframe1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="原始视频", channels="RGB")

        # 显示图像（处理后）
        if processed.ndim == 2:  # 灰度图
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        else:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        stframe2.image(processed, caption="处理后视频", channels="RGB")

        time.sleep(1.0 / fps)

    cap.release()
