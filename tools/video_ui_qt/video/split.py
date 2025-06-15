import cv2
import os

# 输入视频路径
video_path = '1.mp4'  # 替换为你的视频文件路径
# 输出帧图像保存路径
output_dir = 'frames0'
os.makedirs(output_dir, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频")
    exit()

frame_index = 0

print("开始提取帧...")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取完毕

    # 生成图像文件名，例如 frame_0000.jpg
    frame_filename = os.path.join(output_dir, f"frame_{frame_index:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_index += 1

print(f"提取完成，总帧数: {frame_index}")
cap.release()
