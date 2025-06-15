# Car

## 项目概述

智能小车系统是一个融合视觉跟踪与雷达检测的综合性开发项目，主要应用于智能机器人、自动化巡检、光电竞赛等场景。项目通过Python实现核心算法，结合硬件设备实现对目标的精准跟踪与环境感知，具备以下核心能力：

- **视觉跟踪模块**：基于摄像头与PID控制技术，实现目标的水平/垂直跟踪，支持舵机角度实时调整与小车运动控制
- **雷达检测模块**：通过解析雷达数据（角度/距离），完成目标物体的检测、定位与特征分析
- **运动控制模块**：支持麦克纳姆轮全向移动，实现小车平移、旋转等运动模式
- **交互界面模块**：基于PyQt5构建可视化操作界面，支持实时图像显示与参数调节

## 目录结构详解

```
Car/
├── README.md              # 项目总说明文档
├── requirement.txt        # 依赖库清单
├── .gitignore             # Git版本控制忽略规则
├── Track/                 # 核心功能模块
│   ├── lidar/             # 雷达检测子模块
│   │   ├── lidar_scan.py       # 雷达参数配置与初始化
│   │   ├── lidar_scan_plus.py  # 增强版雷达配置（含调试参数）
│   │   ├── t_lidar.py          # 雷达数据处理主逻辑
│   │   ├── t_lidar_plus.py     # 交互式雷达测试工具
│   │   └── lidar_data/        # 雷达数据存储目录
│   │       ├── plot.py         # 雷达数据可视化脚本
│   │       └── *.csv           # 雷达原始数据文件
│   ├── vision/            # 视觉跟踪子模块
│   │   └── vision.py       # 增强型视觉跟踪系统（含PID控制）
│   └── misc/              # 辅助功能脚本
│       └── vision_claude.py  # 基础水平跟踪实现（CLAUDE算法）
├── TrafficLight/          # 交通灯处理模块
│   └── lib/               # 运动控制库
│       └── move.py         # 麦克纳姆轮运动控制核心
└── tools/                 # 工具集
    └── video_ui_qt/       # 视频交互界面
        └── video_general.py  # 视频帧处理与显示逻辑
```

## 环境配置指南

### 软件环境

#### Python依赖安装
```bash
# 确保Python版本为3.9.9
python --version  # 需输出Python 3.9.9

# 安装依赖库
pip install -r requirement.txt
```

#### 核心依赖说明
| 库名             | 版本   | 功能说明                                   |
| ---------------- | ------ | ------------------------------------------ |
| PyQt5            | >=5.15 | 图形界面开发，提供可视化操作面板           |
| opencv-python    | >=4.5  | 计算机视觉处理，支持摄像头图像捕获与分析   |
| numpy            | >=1.21 | 数值计算基础库，用于雷达数据处理与矩阵运算 |
| matplotlib       | >=3.4  | 数据可视化，绘制雷达检测结果图表           |
| ipywidgets       | >=7.6  | Jupyter交互式组件，支持调试时参数动态调节  |
| jupyter notebook | >=6.4  | 交互式开发环境，便于算法调试与结果展示     |

### 硬件环境

#### 必配设备
- **摄像头**：支持cv2.VideoCapture调用，推荐参数：
  - 分辨率：1280×720及以上
  - 帧率：30fps及以上
  - 接口：USB 2.0/3.0

- **舵机**：用于控制摄像头转向，推荐参数：
  - 类型：标准SG90舵机或数字舵机
  - 角度范围：0°~180°
  - 控制信号：PWM信号（频率50Hz）

- **雷达**：用于环境感知，推荐参数：
  - 类型：2D激光雷达（如RPLIDAR A2）
  - 扫描范围：360°
  - 数据输出：角度+距离（CSV格式）

#### 可选设备
- **麦克纳姆轮底盘**：支持全向移动，需匹配电机驱动板
- **树莓派/工控机**：边缘计算设备，推荐Raspberry Pi 4B（4GB+）

## 核心模块详解

### 视觉跟踪模块

#### HorizontalTracker类（基础水平跟踪）
**功能**：实现目标水平方向跟踪，通过PID算法控制舵机转向

**核心属性**：
- `dead_zone`：像素死区（默认10px），减少舵机抖动
- `horizontal_pid`：PID控制器（Kp=0.5, Ki=0.1, Kd=0.05）
- `camera`：OpenCV摄像头对象（默认分辨率640×480）
- `servo_center_angle`：舵机中心角度（默认90°）
- `pid_to_servo_scale`：PID输出到舵机角度映射系数（默认0.8）

**关键方法**：
```python
# 设置目标位置并调整舵机
def set_target_position(target_x):
    """
    参数:
        target_x: 目标在图像中的X坐标（0~640）
    功能:
        计算目标与图像中心误差，通过PID算法生成舵机控制信号
    """
    
# 实时跟踪循环（支持鼠标点击设目标）
def track_with_pid():
    """
    功能:
        启动摄像头实时捕获，通过鼠标点击图像设置跟踪目标
        实时显示跟踪结果与舵机状态
    """
```

#### EnhancedTracker类（增强型跟踪系统）
**功能**：支持水平/垂直双向跟踪，集成小车运动辅助跟踪

**核心属性**：
- `dead_zone_horizontal/vertical`：双轴死区控制
- `horizontal_pid/vertical_pid`：双轴PID控制器
- `robot_rotation_threshold`：小车旋转阈值（默认45°）
- `movement_recorder`：运动轨迹记录器，支持轨迹还原
- `robot_speed`：小车移动速度（默认10cm/s）

**关键方法**：
```python
# 双轴目标跟踪
def set_target_position(target_x, target_y):
    """
    参数:
        target_x/y: 目标在图像中的坐标
    功能:
        同时计算水平/垂直误差，分别控制对应舵机
        当误差超过阈值时触发小车整体旋转
    """
    
# 小车辅助旋转控制
def _rotate_robot_with_advance(direction, angle_deviation):
    """
    参数:
        direction: 旋转方向（'left'/'right'）
        angle_deviation: 偏差角度
    功能:
        控制小车边旋转边前进，同时记录运动轨迹
        用于目标超出摄像头视野时的追踪补偿
    """
```

### 雷达检测模块

#### CarDetector类（小车检测核心）
**功能**：解析雷达数据，检测目标物体并计算位置信息

**核心属性**：
- `distance_threshold`：距离跳变阈值（默认0.3m）
- `min_car_width/max_car_width`：小车宽度过滤范围（默认0.5~2.5m）
- `point_filter_ratio`：无效点过滤比例（默认0.2）

**关键方法**：
```python
# 计算极坐标点距离
def calculate_distance_between_points(r1, angle1, r2, angle2):
    """
    参数:
        r1/r2: 距离值（m）
        angle1/angle2: 角度值（弧度）
    返回:
        两点间欧氏距离（m）
    """
    
# 检测目标物体
def detect_objects(angles, ranges):
    """
    参数:
        angles: 角度数组（弧度）
        ranges: 距离数组（m）
    返回:
        检测到的物体列表，每个物体包含：
        - start_idx: 起始点索引
        - end_idx: 结束点索引
        - width: 物体宽度（m）
        - center_angle: 中心角度（弧度）
        - center_distance: 中心距离（m）
    """
```

#### 雷达数据处理流程
1. 数据加载：从CSV文件读取`angle`和`distance`两列数据
2. 预处理：角度排序、无效点过滤（距离为0或超出雷达范围）
3. 跳变点检测：识别距离突变位置（物体边界）
4. 物体分割：根据跳变点划分独立物体区域
5. 特征计算：计算每个物体的宽度、中心位置等参数
6. 目标筛选：根据宽度阈值过滤无效物体，选择最近目标
7. 结果可视化：绘制雷达点云图与物体边界

### 运动控制模块

#### move.py核心功能
**功能**：控制麦克纳姆轮小车实现全向移动

**支持动作**：
- 左右平移：`move_left(speed)` / `move_right(speed)`
- 前后移动：`move_forward(speed)` / `move_backward(speed)`
- 原地旋转：`rotate_left(speed)` / `rotate_right(speed)`
- 复合运动：`move(x, y, angle)`（同时控制平移与旋转）

**参数说明**：
- `x`：左右位移（正值向右，范围0~100）
- `y`：前后位移（正值向前，范围0~100）
- `angle`：旋转角度（正值顺时针，范围0~360）
- `move_speed`：平移速度（默认100，范围0~100）
- `rotate_speed`：旋转速度（默认100，范围0~100）

## 详细使用指南

### 雷达检测模块使用

#### 1. 准备雷达数据
- 格式要求：CSV文件必须包含`angle`和`distance`列
- 示例数据格式：
  ```csv
  angle,distance
  0.1,2.5
  0.2,2.6
  ...
  ```

#### 2. 运行检测程序
```bash
# 基础检测（输出检测结果到控制台）
python Track/lidar/t_lidar.py --input lidar_data/lidar_data0.csv

# 增强检测（显示可视化界面）
python Track/lidar/t_lidar_plus.py --input lidar_data/lidar_data0.csv --visualize
```

#### 3. 命令行参数说明
| 参数        | 说明                | 默认值         |
| ----------- | ------------------- | -------------- |
| --input     | 雷达数据CSV文件路径 | 无（必须指定） |
| --threshold | 距离跳变阈值（m）   | 0.3            |
| --min-width | 最小物体宽度（m）   | 0.5            |
| --max-width | 最大物体宽度（m）   | 2.5            |
| --visualize | 是否显示可视化界面  | False          |

### 视觉跟踪模块使用

#### 1. 基础水平跟踪测试
```python
from Car.Track.misc.vision_claude import HorizontalTracker
import time

# 初始化跟踪器
tracker = HorizontalTracker(
    dead_zone=10,          # 死区像素
    kp=0.6, ki=0.1, kd=0.05  # PID参数
)

try:
    # 复位舵机到中心位置
    tracker.reset_servo()
    time.sleep(1)
    
    # 启动跟踪
    tracker.start_tracking()
    
    # 测试不同目标位置
    test_positions = [100, 200, 320, 450, 540]  # 图像X坐标
    for target_x in test_positions:
        print(f"测试目标位置: X={target_x}")
        tracker.set_target_position(target_x)
        time.sleep(2)  # 等待舵机稳定
        
except KeyboardInterrupt:
    print("程序中断，正在清理资源...")
finally:
    tracker.cleanup()  # 释放摄像头资源
```

#### 2. 增强型双轴跟踪
```python
from Car.Track.vision.vision import EnhancedTracker

# 初始化增强跟踪器
tracker = EnhancedTracker(
    # 水平方向参数
    horizontal_kp=0.7, horizontal_ki=0.1, horizontal_kd=0.05,
    horizontal_dead_zone=15,
    # 垂直方向参数
    vertical_kp=0.6, vertical_ki=0.05, vertical_kd=0.03,
    vertical_dead_zone=10,
    # 小车运动参数
    robot_rotation_threshold=40,  # 旋转阈值（度）
    robot_speed=80,               # 小车速度（%）
)

# 启动跟踪（假设已连接摄像头和小车）
tracker.start_tracking()

# 设置目标位置（示例：图像中心偏右上）
tracker.set_target_position(400, 150)
```

### 运动控制模块使用

#### 1. 基本运动控制示例
```python
from TrafficLight.lib.move import *

# 向右移动10cm
move_right(100)  # 速度100%
time.sleep(1)    # 移动1秒
stop_robot()     # 停止

# 向前移动并左转
move_forward(80)  # 速度80%
time.sleep(1.5)
rotate_left(50)   # 旋转速度50%
time.sleep(0.8)
stop_robot()

# 复合运动：向右前方移动并右转
move(50, 30, 20)  # x=50, y=30, angle=20
time.sleep(1.2)
stop_robot()
```

#### 2. 轨迹记录与还原
```python
from Car.Track.vision.vision import MovementRecord

# 记录运动轨迹
recorder = MovementRecord()

# 执行一系列动作
recorder.start_recording()
move_forward(100)
time.sleep(1)
rotate_right(70)
time.sleep(0.5)
move_left(80)
time.sleep(0.8)
recorder.stop_recording()

# 还原轨迹（速度为原轨迹的80%）
recorder.playback(speed_ratio=0.8)
```

## 调试与优化指南

### PID参数调试流程
1. **初始化参数**：设置Kp=0.5, Ki=0.1, Kd=0.05（基础值）
2. **调整Kp**：逐渐增加比例系数，直到系统响应迅速但不过度震荡
   - 过小：跟踪缓慢，误差大
   - 过大：舵机震荡，跟踪不稳定
3. **添加Ki**：当Kp调整合适后，增加积分系数消除静态误差
   - 过大：响应滞后，可能引发震荡
4. **调整Kd**：添加微分系数减少震荡，提高稳定性
   - 过大：系统反应迟钝，对快速移动目标跟踪不佳
5. **死区调节**：根据舵机精度设置dead_zone（通常5~15px）

### 雷达数据异常排查
1. **数据格式检查**：
   - 确认CSV文件包含`angle`和`distance`列
   - 检查数据是否有缺失值（使用`nan`或0填充无效点）
   - 确保角度范围在0~2π（弧度）或0~360（度）

2. **检测效果优化**：
   - 调整`distance_threshold`：近距离场景减小阈值（0.2m），远距离增大（0.5m）
   - 修正`min_car_width`和`max_car_width`：根据实际检测目标大小调整
   - 增加`point_filter_ratio`：过滤噪点（建议0.1~0.3）

### 视觉跟踪常见问题
| 问题现象     | 可能原因                         | 解决方案                                            |
| ------------ | -------------------------------- | --------------------------------------------------- |
| 舵机抖动严重 | 死区设置过小或Kp过大             | 增大dead_zone（如15→20），减小Kp（如0.6→0.4）       |
| 跟踪滞后明显 | Kp过小或Ki过大                   | 增大Kp，减小Ki                                      |
| 目标超出视野 | 舵机角度范围不足或小车未辅助移动 | 扩展舵机角度限制，或调高`robot_rotation_threshold`  |
| 图像模糊     | 摄像头帧率不足或焦距未调         | 降低分辨率（如640×480→320×240），手动调节摄像头焦距 |

## 项目扩展建议

### 功能扩展方向
1. **多目标跟踪**：基于YOLO等目标检测模型，实现多目标同时跟踪
2. **路径规划**：结合雷达地图构建，实现自主避障与路径规划
3. **深度学习优化**：将传统PID跟踪升级为深度学习控制（如强化学习）
4. **无线远程控制**：开发手机APP或Web界面，实现远程监控与控制

### 硬件升级方案
1. **传感器升级**：
   - 激光雷达→3D毫米波雷达（增强复杂环境检测能力）
   - 普通摄像头→双目摄像头（增加深度感知）

2. **计算平台升级**：
   - 树莓派→NVIDIA Jetson系列（提升AI计算能力）
   - 增加FPGA加速模块（优化实时性要求高的任务）

3. **运动系统升级**：
   - 麦克纳姆轮→全地形履带（适应复杂地形）
   - 增加机械臂模块（扩展交互能力）

## 贡献指南

### 代码提交规范
1. **分支命名**：
   - 功能开发：`feature/xxx-function`
   - 问题修复：`fix/xxx-bug`
   - 文档更新：`docs/xxx-document`

2. ** commit 信息**：
   ```
   [模块名] 简要描述
   
   详细说明：
   - 变更点1
   - 变更点2
   - ...
   
   关联Issue：#123
   ```

3. **代码风格**：
   - 遵循PEP8规范
   - 函数/类注释使用Google风格
   - 关键逻辑添加行注释

### 问题反馈格式
```
## 问题描述
清晰描述问题现象，如："雷达检测时频繁误报"

## 复现步骤
1. 执行命令：python t_lidar.py ...
2. 操作步骤1
3. 操作步骤2

## 预期结果
期望的正常行为，如："正确检测到小车目标"

## 实际结果
实际出现的异常，如："检测到多个虚假目标"

## 环境信息
- Python版本：3.9.9
- 依赖库版本：如opencv-python=4.5.5
- 硬件型号：雷达型号/Raspberry Pi版本
- 数据文件：可附上测试数据（如lidar_test.csv）
```

## 许可证与声明

本项目遵循[MIT许可证](LICENSE)，允许商业使用、修改和再分发，但需保留原作者声明。

**注意**：硬件连接部分请严格遵循安全规范，强电部分需由专业人员操作。雷达设备使用时需避免直射人眼，摄像头使用需遵守隐私保护法规。# 项目名称：智能小车系统

## 一、项目概述
本项目是一个智能小车系统，专为光电竞赛设计。该系统集成了雷达检测、视觉跟踪、运动控制等多种功能，通过Python代码实现各个模块的逻辑，并利用相关库进行数据处理和界面显示。系统采用模块化设计，各模块之间相互协作，实现小车的自主导航和目标跟踪。

## 二、目录结构
```
Car/
├── README.md              # 项目说明文档
├── requirement.txt        # 项目依赖库列表
├── .gitignore             # Git忽略文件配置
├── Track/                 # 轨迹相关模块
│   ├── lidar/             # 雷达相关代码
│   │   ├── lidar_scan_plus.py  # 增强版雷达参数设置
│   │   ├── lidar_scan.py       # 基础雷达参数设置
│   │   ├── lidar_data/        # 雷达数据存储目录
│   │   │   └── plot.py         # 雷达数据绘图工具
│   │   ├── t_lidar.py          # 雷达数据处理和目标检测
│   │   └── t_lidar_plus.py     # 雷达交互式测试工具
│   ├── vision/            # 视觉相关代码
│   │   └── vision.py       # 视觉跟踪和运动记录
│   └── misc/              # 其他辅助代码
│       └── vision_claude.py  # 水平视觉跟踪
├── TrafficLight/          # 交通灯相关模块
│   └── lib/               # 运动控制库
│       └── move.py         # 麦克纳姆轮运动控制
└── tools/                 # 工具模块
    └── video_ui_qt/       # 视频界面相关代码
        └── video_general.py  # 视频帧处理
```

## 三、环境配置
### 3.1 Python版本
本项目需要Python 3.9.9版本。建议使用虚拟环境来管理项目依赖，避免不同项目之间的依赖冲突。以下是创建和激活虚拟环境的命令：

```bash
# 创建虚拟环境
python3.9 -m venv car_env

# 激活虚拟环境（Windows）
car_env\Scripts\activate

# 激活虚拟环境（Linux/MacOS）
source car_env/bin/activate
```

### 3.2 依赖库安装
在项目根目录下，使用以下命令安装依赖库：
```bash
pip install -r requirement.txt
```
依赖库列表如下：
- PyQt5：用于创建图形用户界面，提供良好的交互体验。
- opencv-python：用于图像处理和视频处理，实现目标检测和跟踪。
- numpy：用于高效的数值计算，处理雷达数据和图像数据。
- matplotlib：用于数据可视化，展示雷达检测结果和运动轨迹。
- ipywidgets：用于创建交互式界面，方便调试和测试。
- ipython：提供增强的交互式Python环境，支持代码自动补全和调试。
- jupyter notebook：用于创建和共享代码、文本和可视化结果，方便项目文档和实验记录。

### 3.3 硬件连接
- **雷达设备**：通过USB串口连接到计算机，默认端口为`/dev/ttyUSB0`（Linux/MacOS）或`COM3`（Windows）。
- **摄像头**：通过USB接口连接到计算机，确保系统能够识别摄像头设备。
- **小车控制板**：通过串口或网络连接到计算机，实现对小车运动的控制。

## 四、模块说明
### 4.1 雷达模块
#### 4.1.1 雷达参数设置
`Track/lidar/lidar_scan_plus.py` 和 `Track/lidar/lidar_scan.py` 文件中定义了 `setup_lidar` 函数，用于设置雷达的各项参数。主要参数包括：
- `RMAX`：最大检测距离，默认32.0米。
- `port`：串口端口，默认`/dev/ttyUSB0`。
- `baudrate`：波特率，默认230400。
- `scanning_frequency`：扫描频率，默认15.0Hz。

```python
def setup_lidar():
    RMAX = 32.0
    port = "/dev/ttyUSB0"
    baudrate = 230400
    scanning_frequency = 15.0
    
    # 创建雷达对象并设置参数
    laser = Lidar(port, baudrate, scanning_frequency)
    laser.start_scan()
    
    return laser
```

#### 4.1.2 雷达数据处理
`Track/lidar/t_lidar.py` 文件实现了雷达数据的处理和目标检测功能。主要功能包括：
1. **数据加载**：从CSV文件或实时雷达设备加载数据。
2. **数据预处理**：对雷达数据进行滤波和排序。
3. **目标检测**：基于距离跳变检测算法识别目标物体。
4. **目标选择**：选择最近的物体作为跟踪目标。
5. **结果可视化**：使用matplotlib绘制雷达点云和检测结果。

```python
class CarDetector:
    def __init__(self):
        self.distance_jump_threshold = 0.3  # 距离跳变阈值
        self.min_car_width = 0.3            # 最小车宽
        self.max_car_width = 2.0            # 最大车宽
        
    def detect_objects(self, angles, ranges):
        """检测雷达数据中的物体"""
        objects = []
        current_object = []
        
        for i in range(len(angles) - 1):
            # 计算当前点与下一点之间的距离差
            range_diff = abs(ranges[i] - ranges[i+1])
            
            # 如果距离差超过阈值，认为是物体边界
            if range_diff > self.distance_jump_threshold:
                if len(current_object) > 5:  # 忽略过小的物体
                    objects.append(current_object)
                current_object = []
            else:
                current_object.append((angles[i], ranges[i]))
        
        # 添加最后一个物体
        if len(current_object) > 5:
            objects.append(current_object)
        
        # 过滤不符合车宽的物体
        filtered_objects = []
        for obj in objects:
            min_angle = min([angle for angle, _ in obj])
            max_angle = max([angle for angle, _ in obj])
            avg_range = sum([r for _, r in obj]) / len(obj)
            width = 2 * avg_range * math.sin(math.radians((max_angle - min_angle) / 2))
            
            if self.min_car_width <= width <= self.max_car_width:
                filtered_objects.append(obj)
                
        return filtered_objects
```

#### 4.1.3 雷达交互式测试
`Track/lidar/t_lidar_plus.py` 文件提供了一个交互式测试工具，支持三种测试模式：
1. **CSV文件测试**：从CSV文件加载雷达数据进行处理和分析。
2. **模拟数据测试**：生成模拟雷达数据，用于算法调试和验证。
3. **实时数据测试**：连接实际雷达设备，实时处理和显示雷达数据。

```python
def interactive_test():
    print("请选择测试模式:")
    print("1. 从CSV文件读取数据")
    print("2. 使用模拟数据测试")
    print("3. 退出")
    
    detector = CarDetector()
    
    while True:
        choice = input("请输入选项 (1-3): ").strip()
        
        if choice == "1":
            csv_file = input("请输入CSV文件路径: ").strip()
            try:
                angles, ranges = load_lidar_data(csv_file)
                process_and_display_data(angles, ranges, detector)
            except FileNotFoundError:
                print(f"错误: 文件 '{csv_file}' 不存在")
                
        elif choice == "2":
            angles, ranges = generate_simulation_data()
            process_and_display_data(angles, ranges, detector)
            
        elif choice == "3":
            print("退出测试")
            break
            
        else:
            print("无效选项，请重新输入")
```

### 4.2 视觉模块
#### 4.2.1 视觉跟踪和运动记录
`Track/vision/vision.py` 文件实现了视觉跟踪和运动记录功能。主要功能包括：
- **目标跟踪**：基于OpenCV的跟踪算法，实现对特定目标的跟踪。
- **运动记录**：记录小车的运动轨迹和状态，支持轨迹回放。
- **舵机控制**：根据目标位置，计算并控制水平和垂直舵机的角度。

```python
class EnhancedTracker:
    def __init__(self):
        # 初始化参数
        self.frame_width = 640
        self.frame_height = 480
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # PID控制器参数
        self.horizontal_pid = PID(Kp=0.5, Ki=0.01, Kd=0.1)
        self.vertical_pid = PID(Kp=0.5, Ki=0.01, Kd=0.1)
        
        # 舵机角度限制
        self.min_horizontal_angle = 30
        self.max_horizontal_angle = 150
        self.min_vertical_angle = 30
        self.max_vertical_angle = 120
        
        # 当前舵机角度
        self.current_horizontal_angle = 90
        self.current_vertical_angle = 90
        
        # 运动记录
        self.movement_record = MovementRecord()
        
    def set_target_position(self, target_x, target_y):
        """设置目标位置并计算控制输出"""
        # 计算误差
        error_x = self.center_x - target_x
        error_y = self.center_y - target_y
        
        # 更新PID控制器
        self.horizontal_pid.SystemOutput = target_x
        self.vertical_pid.SystemOutput = target_y
        
        # 获取PID控制输出
        horizontal_output = self.horizontal_pid.GetPID()
        vertical_output = self.vertical_pid.GetPID()
        
        # 转换为舵机角度调整
        horizontal_adjustment = -horizontal_output * 0.1
        vertical_adjustment = vertical_output * 0.1
        
        # 更新舵机角度
        self.current_horizontal_angle += horizontal_adjustment
        self.current_vertical_angle += vertical_adjustment
        
        # 限制舵机角度范围
        self.current_horizontal_angle = max(
            self.min_horizontal_angle, 
            min(self.max_horizontal_angle, self.current_horizontal_angle)
        )
        self.current_vertical_angle = max(
            self.min_vertical_angle, 
            min(self.max_vertical_angle, self.current_vertical_angle)
        )
        
        # 记录运动
        self.movement_record.add_entry(
            time.time(), 
            target_x, target_y, 
            self.current_horizontal_angle, self.current_vertical_angle
        )
        
        # 控制舵机
        self._control_servos()
        
    def _control_servos(self):
        """控制水平和垂直舵机"""
        # 这里是实际控制舵机的代码
        # 在开发环境中，我们只打印舵机角度
        print(f"水平舵机角度: {self.current_horizontal_angle:.1f}°")
        print(f"垂直舵机角度: {self.current_vertical_angle:.1f}°")
```

#### 4.2.2 水平视觉跟踪
`Track/misc/vision_claude.py` 文件专门实现了水平视觉跟踪功能。该模块使用PID控制器来稳定跟踪目标，具有以下特点：
- 基于颜色识别的目标检测。
- PID控制器实现平滑跟踪。
- 提供固定位置测试和实时跟踪测试两种模式。

```python
class HorizontalTracker:
    def __init__(self):
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 初始化PID控制器
        self.horizontal_pid = PID(
            SetPoint=320,  # 图像中心x坐标
            Kp=0.5, Ki=0.01, Kd=0.1
        )
        
        # 舵机控制参数
        self.pid_to_servo_scale = 0.2  # PID输出到舵机角度的缩放因子
        self.current_servo_angle = 90  # 初始舵机角度
        
        # 颜色检测参数
        self.lower_color = np.array([20, 100, 100])  # HSV颜色下界
        self.upper_color = np.array([40, 255, 255])  # HSV颜色上界
        
    def detect_target(self, frame):
        """检测目标物体并返回其中心坐标"""
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建颜色掩码
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        # 进行形态学操作，消除噪声
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 找到最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓的矩
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                # 计算目标中心
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 在图像上绘制目标中心和轮廓
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                
                return cx, cy, frame
                
        return None, None, frame
        
    def track_target(self):
        """实时跟踪目标"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 检测目标
            target_x, target_y, frame = self.detect_target(frame)
            
            if target_x is not None:
                # 设置PID目标值
                self.horizontal_pid.SystemOutput = target_x
                
                # 计算PID输出
                pid_adjustment = self.horizontal_pid.GetPID()
                
                # 转换为舵机角度调整
                angle_adjustment = -pid_adjustment * self.pid_to_servo_scale
                
                # 更新舵机角度
                self.current_servo_angle += angle_adjustment
                
                # 限制舵机角度范围
                self.current_servo_angle = max(30, min(150, self.current_servo_angle))
                
                # 显示跟踪信息
                cv2.putText(frame, f"Target X: {target_x}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Servo Angle: {self.current_servo_angle:.1f}°", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 绘制目标区域和当前角度指示
                cv2.line(frame, (target_x, 0), (target_x, 480), (0, 0, 255), 2)
                cv2.line(frame, (320, 0), (320, 480), (255, 0, 0), 2)
                
            # 显示图像
            cv2.imshow('Horizontal Tracking', frame)
            
            # 按ESC键退出
            if cv2.waitKey(1) == 27:
                break
                
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
```

### 4.3 运动控制模块
`TrafficLight/lib/move.py` 文件实现了麦克纳姆轮的运动控制功能。麦克纳姆轮具有全向移动能力，可以实现前后、左右和旋转运动。该模块提供了以下主要功能：

```python
def move(x, y, angle, move_speed=100, rotate_speed=100):
    """
    控制麦克纳姆轮小车运动
    
    参数:
    x: 左右移动距离（负值表示左移，正值表示右移）
    y: 前后移动距离（负值表示后退，正值表示前进）
    angle: 旋转角度（负值表示左转，正值表示右转）
    move_speed: 移动速度（0-255）
    rotate_speed: 旋转速度（0-255）
    """
    try:
        # 初始化GPIO
        GPIO.setmode(GPIO.BCM)
        
        # 设置电机控制引脚
        motor_pins = [17, 18, 22, 23]  # 假设四个电机的控制引脚
        for pin in motor_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
            
        # 控制左右移动
        if x != 0:
            if x < 0:
                # 左移
                print(f"左移 {abs(x)} 个单位")
                move_left(move_speed)
            else:
                # 右移
                print(f"右移 {x} 个单位")
                move_right(move_speed)
                
            # 移动指定时间
            time.sleep(abs(x) / move_speed)
            stop_robot()
            
        # 控制前后移动
        if y != 0:
            if y < 0:
                # 后退
                print(f"后退 {abs(y)} 个单位")
                move_backward(move_speed)
            else:
                # 前进
                print(f"前进 {y} 个单位")
                move_forward(move_speed)
                
            # 移动指定时间
            time.sleep(abs(y) / move_speed)
            stop_robot()
            
        # 控制旋转
        if angle != 0:
            if angle < 0:
                # 左转
                print(f"左转 {abs(angle)} 度")
                rotate_left(rotate_speed)
            else:
                # 右转
                print(f"右转 {angle} 度")
                rotate_right(rotate_speed)
                
            # 旋转指定时间
            time.sleep(abs(angle) / (rotate_speed * 0.5))  # 0.5是经验系数，需根据实际情况调整
            stop_robot()
            
    except KeyboardInterrupt:
        # 用户按下Ctrl+C时停止机器人
        stop_robot()
        print("操作已取消")
    finally:
        # 清理GPIO资源
        GPIO.cleanup()

# 辅助函数：控制四个电机实现不同运动
def move_forward(speed):
    # 前两个电机正转，后两个电机反转
    set_motor_speeds(speed, speed, -speed, -speed)
    
def move_backward(speed):
    # 前两个电机反转，后两个电机正转
    set_motor_speeds(-speed, -speed, speed, speed)
    
def move_left(speed):
    # 左移：左上和右下电机正转，右上和左下电机反转
    set_motor_speeds(speed, -speed, -speed, speed)
    
def move_right(speed):
    # 右移：左上和右下电机反转，右上和左下电机正转
    set_motor_speeds(-speed, speed, speed, -speed)
    
def rotate_left(speed):
    # 左转：所有电机反转
    set_motor_speeds(-speed, -speed, -speed, -speed)
    
def rotate_right(speed):
    # 右转：所有电机正转
    set_motor_speeds(speed, speed, speed, speed)
    
def stop_robot():
    # 停止所有电机
    set_motor_speeds(0, 0, 0, 0)
    
def set_motor_speeds(motor1, motor2, motor3, motor4):
    """设置四个电机的速度"""
    # 这里是实际控制电机的代码
    # 在开发环境中，我们只打印电机速度
    print(f"电机速度: M1={motor1}, M2={motor2}, M3={motor3}, M4={motor4}")
```

### 4.4 视频界面模块
`tools/video_ui_qt/video_general.py` 文件实现了视频帧的处理和显示功能。该模块基于PyQt5创建了一个视频播放界面，支持以下功能：
- 视频帧的实时显示
- 视频录制功能
- 视频帧的简单处理
- 界面交互控制

```python
class VideoPlayer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题和大小
        self.setWindowTitle("视频播放器")
        self.resize(800, 600)
        
        # 创建中央部件和布局
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # 创建视频显示区域
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.video_label)
        
        # 创建控制按钮
        self.control_layout = QtWidgets.QHBoxLayout()
        
        self.play_button = QtWidgets.QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_play)
        self.control_layout.addWidget(self.play_button)
        
        self.record_button = QtWidgets.QPushButton("录制")
        self.record_button.clicked.connect(self.toggle_record)
        self.control_layout.addWidget(self.record_button)
        
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.clicked.connect(self.stop)
        self.control_layout.addWidget(self.stop_button)
        
        self.layout.addLayout(self.control_layout)
        
        # 初始化视频捕获和录制
        self.cap = None
        self.recording = False
        self.out = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        
        # 状态变量
        self.playing = False
        self.locked_frame = None
        
    def open_camera(self, camera_index=0):
        """打开摄像头"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "错误", "无法打开摄像头")
            return False
            
        # 设置视频分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        return True
        
    def toggle_play(self):
        """切换播放/暂停状态"""
        if not self.cap:
            if not self.open_camera():
                return
                
        if self.playing:
            self.timer.stop()
            self.play_button.setText("播放")
        else:
            self.timer.start(30)  # 30ms间隔，约33fps
            self.play_button.setText("暂停")
            
        self.playing = not self.playing
        
    def toggle_record(self):
        """切换录制状态"""
        if not self.playing:
            QtWidgets.QMessageBox.warning(self, "警告", "请先开始播放视频")
            return
            
        if self.recording:
            # 停止录制
            if self.out:
                self.out.release()
                self.out = None
                
            self.record_button.setText("录制")
        else:
            # 开始录制
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            self.record_button.setText("停止录制")
            
        self.recording = not self.recording
        
    def stop(self):
        """停止播放和录制"""
        if self.timer.isActive():
            self.timer.stop()
            self.playing = False
            self.play_button.setText("播放")
            
        if self.recording:
            if self.out:
                self.out.release()
                self.out = None
                
            self.recording = False
            self.record_button.setText("录制")
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # 清空显示
        self.video_label.clear()
        
    def next_frame(self):
        """获取下一帧并显示"""
        if not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            # 如果读取失败，尝试重置视频流
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
            
        # 保存当前帧的副本
        self.locked_frame = frame.copy()
        
        # 显示帧
        self.show_frame(frame)
        
        # 如果正在录制，写入当前帧
        if self.recording and self.out is not None:
            self.out.write(frame)
            
    def show_frame(self, frame):
        """显示帧到界面上"""
        # 转换BGR为RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 创建QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        # 调整图像大小以适应标签
        scaled_img = q_img.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio)
        
        # 显示图像
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(scaled_img))
```

## 五、使用方法
### 5.1 雷达数据处理
运行 `Track/lidar/t_lidar.py` 文件，传入CSV文件路径作为参数，即可处理雷达数据并检测小车。

```bash
python Track/lidar/t_lidar.py lidar_data/lidar_data0.csv
```

### 5.2 雷达交互式测试
运行 `Track/lidar/t_lidar_plus.py` 文件，根据提示选择测试模式。

```bash
python Track/lidar/t_lidar_plus.py
```

### 5.3 水平视觉跟踪测试
运行 `Track/misc/vision_claude.py` 文件，进行水平视觉跟踪的固定位置测试和实时跟踪测试。

```bash
python Track/misc/vision_claude.py
```

### 5.4 运动控制
在代码中调用 `TrafficLight/lib/move.py` 文件中的 `move` 函数，传入相应的参数，即可控制小车的运动。

```python
from TrafficLight.lib.move import move

# 示例：向前移动10个单位，右转90度，再向左移动5个单位
move(0, 10, 0)    # 前进
time.sleep(1)     # 等待1秒
move(0, 0, 90)    # 右转
time.sleep(1)     # 等待1秒
move(-5, 0, 0)    # 左移
```

### 5.5 视频界面
运行 `tools/video_ui_qt/video_general.py` 文件，打开视频播放界面。

```bash
python tools/video_ui_qt/video_general.py
```

使用界面上的按钮可以控制视频的播放、录制和停止。

## 六、系统集成
要实现完整的智能小车功能，需要将各个模块集成在一起。以下是一个简单的集成示例：

```python
import time
from Track.lidar.t_lidar import CarDetector, load_lidar_data, process_lidar_data
from Track.vision.vision import EnhancedTracker
from TrafficLight.lib.move import move

def main():
    # 初始化雷达检测器
    radar_detector = CarDetector()
    
    # 初始化视觉跟踪器
    vision_tracker = EnhancedTracker()
    
    try:
        while True:
            # 1. 雷达检测目标
            # 注意：实际应用中应从实时雷达获取数据，这里使用示例数据
            angles, ranges = load_lidar_data("Track/lidar/lidar_data/lidar_data0.csv")
            detected_objects, target = process_lidar_data(angles, ranges, radar_detector)
            
            if target:
                print(f"雷达检测到目标：距离={target[0]:.2f}米, 角度={target[1]:.2f}度")
                
                # 2. 根据雷达目标位置，视觉系统准备跟踪
                # 这里简化处理，假设视觉系统已经启动并开始跟踪
                
                # 3. 根据目标位置计算移动指令
                distance, angle = target
                
                # 如果目标太远，前进
                if distance > 1.0:
                    forward_distance = min(distance - 0.8, 0.5)  # 前进到距离目标0.8米处
                    move(0, forward_distance, 0)
                    time.sleep(1)  # 等待移动完成
                    
                # 如果目标角度偏离中心，调整方向
                if abs(angle) > 5:  # 角度偏差超过5度
                    rotate_angle = min(angle, 30)  # 最大旋转30度
                    move(0, 0, rotate_angle)
                    time.sleep(0.5)  # 等待旋转完成
                    
                # 4. 更新视觉跟踪器
                # 在实际应用中，这里应该获取视觉目标位置
                # 这里简化处理，使用雷达目标角度估计视觉目标位置
                target_x = 320 + angle * 3  # 简单映射：1度对应3个像素
                target_y = 240
                vision_tracker.set_target_position(target_x, target_y)
                
            else:
                print("未检测到目标，正在搜索...")
                # 没有检测到目标，小车旋转搜索
                move(0, 0, 30)  # 右转30度
                time.sleep(1)
                
            # 循环间隔
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("程序已停止")
    finally:
        # 清理资源
        pass

if __name__ == "__main__":
    main()
```

## 七、注意事项
- 请确保Python版本为3.9.9，并正确安装了所有依赖库。
- 在使用雷达模块时，需要连接正确的雷达设备，并确保串口端口和波特率设置正确。
- 在使用视觉模块时，需要连接摄像头，并确保摄像头正常工作。
- 在进行运动控制时，需要确保小车的硬件连接正常，并根据实际情况调整运动速度和时间。
- 运行代码前，请检查GPIO引脚配置，避免与其他设备冲突。
- 在实际部署前，建议在仿真环境中测试各个模块的功能。

## 八、调试和优化
### 8.1 调试工具
- **雷达数据可视化**：使用 `Track/lidar/lidar_data/plot.py` 工具可以可视化雷达数据，帮助调试目标检测算法。
- **日志输出**：各模块都提供了详细的日志输出，可以通过打印信息了解程序运行状态。
- **交互式测试**：利用各模块提供的交互式测试功能，可以单独测试各个组件的功能。

### 8.2 参数优化
- **雷达检测参数**：可以调整 `CarDetector` 类中的 `distance_jump_threshold`、`min_car_width` 和 `max_car_width` 参数，优化目标检测效果。
- **PID控制参数**：可以调整 `EnhancedTracker` 和 `HorizontalTracker` 类中的PID控制器参数（Kp、Ki、Kd），改善跟踪稳定性。
- **运动控制参数**：可以调整 `move` 函数中的速度参数和时间系数，优化小车运动性能。
