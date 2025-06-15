# Car 项目文档

## 项目概述
本项目聚焦于智能小车相关的开发，融合了视觉跟踪与雷达检测两大核心功能。视觉跟踪模块借助摄像头与PID控制技术，实现对目标的精准追踪，同时能控制舵机和小车的运动；雷达检测模块则利用雷达数据，完成对小车的检测与定位。项目适用于智能机器人、自动化巡检等领域，为目标跟踪与环境感知提供了有效的解决方案。

## 环境要求
### 软件环境
- **Python 版本**：Python == 3.9.9
- **依赖库**：请参考 `Car/requirement.txt` 文件，其中包含了项目所需的所有 Python 库，以下是具体说明：
    - `PyQt5`：用于创建图形用户界面（GUI），为项目提供可视化操作界面。
    - `opencv-python`：计算机视觉库，用于处理摄像头图像和视频流，实现目标检测和跟踪功能。
    - `numpy`：用于数值计算和数组处理，在雷达数据处理和图像处理中发挥重要作用。
    - `matplotlib`：用于数据可视化，将雷达检测结果以图形的形式展示出来。
    - `ipywidgets` 和 `ipython`：用于在 Jupyter Notebook 中创建交互式界面，方便进行代码调试和实验。
    - `jupyter notebook`：提供一个交互式的开发环境，方便用户编写和运行代码，查看实验结果。

### 硬件环境
- **摄像头**：用于视觉跟踪功能，需支持 `cv2.VideoCapture` 调用。建议选择分辨率较高、帧率稳定的摄像头，以确保图像质量和跟踪效果。
- **舵机**：用于控制摄像头或小车的转向，需根据实际硬件调整舵机控制库。不同型号的舵机可能具有不同的控制参数和接口，需要根据具体情况进行配置。
- **雷达**：用于雷达检测功能，需能够输出角度和距离数据，并保存为 CSV 文件。雷达的性能和精度会直接影响检测结果，建议选择高精度、高分辨率的雷达设备。

## 目录结构
```plaintext
Car/
├── requirement.txt            # 项目依赖库列表
├── README.md                  # 项目说明文档
├── Track/                     # 核心功能模块
│   ├── misc/                  # 辅助功能文件
│   │   ├── vision_claude.py   # 水平跟踪系统实现
│   │   └── vision.py          # 水平跟踪系统另一种实现
│   ├── lidar/                 # 雷达检测模块
│   │   ├── t_lidar_plus.py    # 分析 segment 特征分布
│   │   ├── t_lidar.py         # 小车检测与数据处理
│   │   └── lidar_scan.py      # 雷达检测结果可视化
│   └── vision/                # 视觉跟踪模块
│       └── vision.py          # 增强跟踪系统实现
└── tools/                     # 工具模块
    ├── video_ui_qt/           # 视频 UI 界面
    └── control_ui_qt/         # 控制 UI 界面
```

## 功能模块说明
### 视觉跟踪模块
#### HorizontalTracker 类
- **功能**：实现水平跟踪功能，包含摄像头初始化、PID 控制、舵机控制等功能。通过设置目标在图像中的 X 坐标位置，利用 PID 算法计算误差并调整舵机角度，使摄像头对准目标。
- **主要属性**：
    - `dead_zone`：像素死区，用于减少舵机的抖动。
    - `horizontal_pid`：PID 控制器，用于计算舵机的角度调整量。
    - `camera`：摄像头对象，用于获取图像帧。
    - `image_width` 和 `image_center_x`：图像的宽度和中心位置。
    - `servo_center_angle`、`servo_min_angle` 和 `servo_max_angle`：舵机的中心角度、最小角度和最大角度。
    - `current_servo_angle`：当前舵机的角度。
    - `pid_to_servo_scale`：PID 输出到舵机角度的映射比例。
- **主要方法**：
    - `set_target_position(target_x)`：设置目标在图像中的 X 坐标位置，并根据误差调整舵机角度。
    - `control_servo(angle)`：控制舵机转动到指定角度。
    - `reset_servo()`：将舵机复位到中心位置。
    - `start_tracking()` 和 `stop_tracking()`：开始和停止跟踪功能。
    - `get_camera_frame()`：获取摄像头的当前图像帧。
    - `draw_tracking_info(frame, target_x=None)`：在图像上绘制跟踪信息，如中心线、死区范围和目标位置。
    - `cleanup()`：清理资源，释放摄像头并关闭所有窗口。
    - `track_with_pid()`：实时跟踪循环，可通过鼠标点击设置目标位置。

#### EnhancedTracker 类
- **功能**：增强跟踪系统，支持水平和垂直方向的跟踪，同时增加了小车旋转辅助跟踪功能。通过记录小车的运动轨迹，可实现反向运动还原。
- **主要属性**：
    - `dead_zone_horizontal` 和 `dead_zone_vertical`：水平和垂直方向的像素死区。
    - `tracking_active`：跟踪状态标志。
    - `horizontal_pid` 和 `vertical_pid`：水平和垂直方向的 PID 控制器。
    - `horizontal_pid_scale` 和 `vertical_pid_scale`：PID 输出到舵机角度的映射比例。
    - `robot_rotation_threshold`：舵机角度超过此值时开始旋转小车。
    - `robot_speed`：小车的运动速度。
    - `rotation_time_per_degree`：每度旋转需要的时间。
    - `camera`：摄像头对象，用于获取图像帧。
    - `image_width`、`image_height`、`image_center_x` 和 `image_center_y`：图像的宽度、高度和中心位置。
    - `horizontal_servo_center`、`horizontal_servo_min` 和 `horizontal_servo_max`：水平舵机的中心角度、最小角度和最大角度。
    - `vertical_servo_center`、`vertical_servo_min` 和 `vertical_servo_max`：垂直舵机的中心角度、最小角度和最大角度。
    - `current_horizontal_angle` 和 `current_vertical_angle`：当前水平和垂直舵机的角度。
    - `movement_recorder`：运动记录对象，用于记录小车的运动轨迹。
    - `movement_lock`：线程锁，用于防止同时执行多个运动指令。
- **主要方法**：
    - `set_target_position(target_x, target_y)`：设置目标在图像中的 X 和 Y 坐标位置，并进行跟踪控制。
    - `_control_horizontal(target_x, error_x)`：水平方向的控制逻辑，根据误差调整水平舵机的角度。
    - `_control_vertical(target_y, error_y)`：垂直方向的控制逻辑，根据误差调整垂直舵机的角度。
    - `_check_robot_rotation()`：检查是否需要旋转小车来辅助跟踪。
    - `_rotate_robot_with_advance(direction, angle_deviation)`：旋转小车并前进，同时记录运动轨迹。

### 雷达检测模块
#### CarDetector 类
- **功能**：实现小车检测功能，通过分析雷达数据中的距离跳变点，检测出可能的小车目标，并计算其位置、角度和宽度等信息。
- **主要属性**：
    - `distance_threshold`：距离跳变阈值，用于判断是否为跳变点。
    - `min_car_width` 和 `max_car_width`：小车的最小和最大宽度，用于筛选合理的小车目标。
- **主要方法**：
    - `calculate_distance_between_points(r1, angle1, r2, angle2)`：计算两个极坐标点之间的欧几里得距离。
    - `find_jump_points(angles, ranges)`：找到距离跳变点的索引和跳变类型。
    - `detect_objects(angles, ranges)`：检测目标物体，返回检测到的物体列表。
    - `get_target_position(detected_objects)`：从检测到的物体中选择目标（通常选择最近的），返回目标物体的信息。

#### analyze_segment_features 函数
- **功能**：分析雷达检测结果中 segment 特征的分布，包括长度、距离、拟合误差和点数等统计信息。

#### visualize_detection 函数
- **功能**：将雷达检测结果可视化，以散点图和边界线的形式展示雷达点和检测到的物体。

## 使用方法
### 视觉跟踪模块
```python
from Car.Track.misc.vision import HorizontalTracker

# 创建跟踪器
tracker = HorizontalTracker()

try:
    # 复位舵机
    tracker.reset_servo()
    time.sleep(1)

    # 开始跟踪
    tracker.start_tracking()

    # 模拟不同的目标位置进行测试
    test_positions = [100, 200, 320, 450, 540]  # X 坐标位置
    for target_x in test_positions:
        print(f"\n测试目标位置: X={target_x}")
        tracker.set_target_position(target_x)
        time.sleep(1)  # 等待舵机调整

except KeyboardInterrupt:
    print("\n程序被用户中断")

finally:
    # 清理资源
    tracker.cleanup()
```

### 雷达检测模块
```python
from Car.Track.lidar.t_lidar import process_lidar_data

# 处理雷达数据并检测小车
csv_file_path = 'path/to/your/lidar_data.csv'
detected_objects, target = process_lidar_data(csv_file_path)
```

## UI 说明
### [video_ui_qt](tools/video_ui_qt) 
该模块提供了一个视频 UI 界面，包含两个用于显示图像的控件。左边的控件默认显示原图，右边的控件可以通过右上角的选择控件自由切换显示内容。上方预留了 `process_frame` 函数处理接口，可加入自定义的图像处理逻辑，以便与原图进行对比查看。
```python
# -----------------保留的函数处理接口--------------------
def process_frame(frame):
    processed = frame.copy()
    # 在这里添加你的图像处理逻辑
    return processed
```

## 注意事项
- **PID 参数调整**：PID 控制器的参数（如比例、积分、微分系数）需要根据实际硬件和应用场景进行调整，以达到最佳的跟踪效果。不同的硬件设备和环境条件可能会导致 PID 参数的最优值不同，建议通过实验和调试来确定合适的参数。
- **舵机控制**：舵机控制部分的代码需要根据实际使用的硬件进行调整，确保舵机能正常工作。不同型号的舵机可能具有不同的控制信号格式和范围，需要根据具体情况进行配置。
- **雷达数据格式**：雷达数据文件必须包含 `angle` 和 `distance` 两列，且数据格式为 CSV。在使用雷达数据之前，请确保数据文件的格式正确，否则可能会导致数据加载失败或检测结果不准确。
