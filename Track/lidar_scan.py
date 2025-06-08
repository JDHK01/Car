import os
import ydlidar
import time
import sys
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math


class CarDetector:
    def __init__(self, distance_threshold=0.3, min_car_width=0.08, max_car_width=0.8):
        """
        初始化小车检测器

        Args:
            distance_threshold: 相邻点距离跳变阈值(米)
            min_car_width: 小车最小宽度(米)
            max_car_width: 小车最大宽度(米)
        """
        self.distance_threshold = distance_threshold
        self.min_car_width = min_car_width
        self.max_car_width = max_car_width

    def calculate_distance_between_points(self, r1, angle1, r2, angle2):
        """
        计算两个极坐标点之间的距离

        Args:
            r1, angle1: 第一个点的极坐标
            r2, angle2: 第二个点的极坐标

        Returns:
            两点之间的欧几里得距离
        """
        x1 = r1 * math.cos(angle1)
        y1 = r1 * math.sin(angle1)
        x2 = r2 * math.cos(angle2)
        y2 = r2 * math.sin(angle2)

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_jump_points(self, angles, ranges):
        """
        找到距离跳变点

        Args:
            angles: 角度列表
            ranges: 距离列表

        Returns:
            jump_indices: 跳变点的索引列表
            jump_types: 跳变类型列表 ('near_to_far' 或 'far_to_near')
        """
        jump_indices = []
        jump_types = []

        for i in range(len(ranges) - 1):
            # 计算相邻两点的距离差
            distance_diff = abs(ranges[i + 1] - ranges[i])

            # 计算两点在空间中的实际距离
            spatial_distance = self.calculate_distance_between_points(
                ranges[i], angles[i], ranges[i + 1], angles[i + 1]
            )

            # 如果距离差或空间距离超过阈值，认为是跳变
            if distance_diff > self.distance_threshold or spatial_distance > self.min_car_width:
                jump_indices.append(i)

                # 判断跳变类型
                if ranges[i + 1] > ranges[i]:
                    jump_types.append('near_to_far')  # 从近到远
                else:
                    jump_types.append('far_to_near')  # 从远到近

        return jump_indices, jump_types

    def detect_objects(self, angles, ranges):
        """
        检测目标物体

        Args:
            angles: 角度列表
            ranges: 距离列表

        Returns:
            detected_objects: 检测到的物体列表，每个物体包含左右边界信息
        """
        jump_indices, jump_types = self.find_jump_points(angles, ranges)

        detected_objects = []

        # 寻找成对的跳变点（一个near_to_far后跟一个far_to_near）
        i = 0

        while i < len(jump_indices) - 1:
            current_idx = jump_indices[i]
            current_type = jump_types[i]

            # 寻找near_to_far类型的跳变
            if current_type == 'near_to_far':
                # 寻找后续的far_to_near跳变
                for j in range(i + 1, len(jump_indices)):
                    next_idx = jump_indices[j]
                    next_type = jump_types[j]

                    if next_type == 'far_to_near':
                        # 找到了一对跳变点，可能是一个物体
                        left_boundary_idx = current_idx
                        right_boundary_idx = next_idx + 1

                        # 计算物体的角度范围和距离
                        left_angle = angles[left_boundary_idx]
                        right_angle = angles[right_boundary_idx]

                        # 计算物体中心区域的平均距离
                        center_start = left_boundary_idx + 1
                        center_end = right_boundary_idx

                        if center_end > center_start:
                            center_ranges = ranges[center_start:center_end]
                            avg_distance = np.mean(center_ranges)

                            # 计算物体的角度宽度和空间宽度
                            angle_width = abs(right_angle - left_angle)
                            spatial_width = avg_distance * angle_width

                            # 验证是否为合理的小车尺寸
                            if self.min_car_width <= spatial_width <= self.max_car_width:
                                detected_objects.append({
                                    'left_boundary_idx': left_boundary_idx,
                                    'right_boundary_idx': right_boundary_idx,
                                    'left_angle': left_angle,
                                    'right_angle': right_angle,
                                    'center_angle': (left_angle + right_angle) / 2,
                                    'distance': avg_distance,
                                    'angular_width': angle_width,
                                    'spatial_width': spatial_width,
                                    'center_ranges': center_ranges
                                })

                        i = j  # 跳到下一个far_to_near之后
                        break
                else:
                    i += 1
            else:
                i += 1

        return detected_objects

    def get_target_position(self, detected_objects):
        """
        从检测到的物体中选择目标（通常选择最近的）

        Args:
            detected_objects: 检测到的物体列表

        Returns:
            target_info: 目标物体的信息，如果没有检测到则返回None
        """
        if not detected_objects:
            return None

        # 选择距离最近的物体作为目标
        target = min(detected_objects, key=lambda obj: obj['distance'])

        return {
            'distance': target['distance'],
            'angle': target['center_angle'],
            'width': target['spatial_width'],
            'left_angle': target['left_angle'],
            'right_angle': target['right_angle']
        }


# 主要的雷达检测函数
def setup_lidar():
    """设置雷达参数"""
    RMAX = 32.0
    port = "/dev/ttyUSB0"

    laser = ydlidar.CYdLidar()
    laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 6.0)
    laser.setlidaropt(ydlidar.LidarPropSampleRate, 3)
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
    laser.setlidaropt(ydlidar.LidarPropAbnormalCheckCount, 4)
    laser.setlidaropt(ydlidar.LidarPropIntenstiyBit, 10)
    laser.setlidaropt(ydlidar.LidarPropIntenstiy, False)
    laser.setlidaropt(ydlidar.LidarPropFixedResolution, False)
    laser.setlidaropt(ydlidar.LidarPropReversion, False)
    laser.setlidaropt(ydlidar.LidarPropInverted, False)
    laser.setlidaropt(ydlidar.LidarPropAutoReconnect, True)
    laser.setlidaropt(ydlidar.LidarPropSupportMotorDtrCtrl, True)
    laser.setlidaropt(ydlidar.LidarPropSupportHeartBeat, False)
    laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
    laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
    laser.setlidaropt(ydlidar.LidarPropMaxRange, 8.0)
    laser.setlidaropt(ydlidar.LidarPropMinRange, 0.1)

    return laser


def main():
    """主函数 - 实时检测小车"""
    # 初始化雷达
    laser = setup_lidar()
    scan = ydlidar.LaserScan()

    # 初始化小车检测器
    detector = CarDetector(
        distance_threshold=0.3,  # 距离跳变阈值
        min_car_width=0.1,  # 小车最小宽度
        max_car_width=1.0  # 小车最大宽度
    )

    # 启动雷达
    ret = laser.initialize()
    if ret:
        ret = laser.turnOn()

        try:
            scan_count = 0
            while ret:
                angles = []
                ranges = []

                # 获取扫描数据
                laser.doProcessSimple(scan)
                for point in scan.points:
                    angles.append(point.angle)
                    ranges.append(point.range)

                if len(angles) > 0:
                    # 排序数据（按角度）
                    sorted_data = sorted(zip(angles, ranges))
                    angles, ranges = zip(*sorted_data)
                    angles = list(angles)
                    ranges = list(ranges)

                    # 检测小车
                    detected_objects = detector.detect_objects(angles, ranges)
                    target = detector.get_target_position(detected_objects)

                    scan_count += 1
                    print(f"\n=== 扫描 #{scan_count} ===")

                    if target:
                        print(f"检测到目标小车!")
                        print(f"  距离: {target['distance']:.2f} 米")
                        print(f"  角度: {math.degrees(target['angle']):.1f} 度")
                        print(f"  宽度: {target['width']:.2f} 米")
                        print(f"  左边界角度: {math.degrees(target['left_angle']):.1f} 度")
                        print(f"  右边界角度: {math.degrees(target['right_angle']):.1f} 度")
                    else:
                        print("未检测到目标小车")

                    # 显示所有检测到的物体
                    if detected_objects:
                        print(f"总共检测到 {len(detected_objects)} 个物体:")
                        for i, obj in enumerate(detected_objects):
                            print(f"  物体 {i + 1}: 距离={obj['distance']:.2f}m, "
                                  f"角度={math.degrees(obj['center_angle']):.1f}°, "
                                  f"宽度={obj['spatial_width']:.2f}m")

                time.sleep(0.1)  # 控制扫描频率

        except KeyboardInterrupt:
            print("\n检测停止")
        finally:
            laser.turnOff()

    laser.disconnecting()


# 可视化函数（可选）
def visualize_detection(angles, ranges, detected_objects):
    """可视化检测结果"""
    plt.figure(figsize=(10, 8))

    # 转换为笛卡尔坐标
    x = [r * math.cos(a) for r, a in zip(ranges, angles)]
    y = [r * math.sin(a) for r, a in zip(ranges, angles)]

    # 绘制所有点
    plt.scatter(x, y, c='blue', s=1, alpha=0.6, label='雷达点')

    # 绘制检测到的物体
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, obj in enumerate(detected_objects):
        color = colors[i % len(colors)]

        # 绘制物体中心
        center_x = obj['distance'] * math.cos(obj['center_angle'])
        center_y = obj['distance'] * math.sin(obj['center_angle'])
        plt.scatter(center_x, center_y, c=color, s=100, marker='x',
                    label=f'物体 {i + 1}')

        # 绘制边界线
        left_x = obj['distance'] * math.cos(obj['left_angle'])
        left_y = obj['distance'] * math.sin(obj['left_angle'])
        right_x = obj['distance'] * math.cos(obj['right_angle'])
        right_y = obj['distance'] * math.sin(obj['right_angle'])

        plt.plot([0, left_x], [0, left_y], '--', color=color, alpha=0.7)
        plt.plot([0, right_x], [0, right_y], '--', color=color, alpha=0.7)

    plt.xlabel('X (米)')
    plt.ylabel('Y (米)')
    plt.title('雷达小车检测结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    main()