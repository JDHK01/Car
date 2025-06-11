import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

plt.rcParams['font.family'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
class CarDetector:
    def __init__(self, distance_threshold=0.3, min_car_width=0.08, max_car_width=0.8):

        """
        =====================功能展示，不做实体============================
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


def load_lidar_data(csv_file_path):
    """
    从CSV文件加载雷达数据

    Args:
        csv_file_path: CSV文件路径

    Returns:
        angles: 角度数组
        ranges: 距离数组
    """
    try:
        # 加载数据
        df = pd.read_csv(csv_file_path)

        # 提取 angle 列和 distance 列的数据到数组中
        angles = np.array(df['angle'])
        ranges = np.array(df['distance'])

        print(f"成功加载数据: {len(angles)} 个数据点")
        print(f"角度范围: {np.min(angles):.2f} 到 {np.max(angles):.2f} 弧度")
        print(f"距离范围: {np.min(ranges):.2f} 到 {np.max(ranges):.2f} 米")

        return angles, ranges

    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file_path}")
        return None, None
    except KeyError as e:
        print(f"错误: CSV文件中缺少列 {e}")
        print("请确保CSV文件包含 'angle' 和 'distance' 列")
        return None, None
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return None, None


def process_lidar_data(csv_file_path):
    """
    处理雷达数据并检测小车

    Args:
        csv_file_path: CSV文件路径
    """
    # 加载数据
    angles, ranges = load_lidar_data(csv_file_path)

    if angles is None or ranges is None:
        print("数据加载失败，程序退出")
        return

    # 初始化小车检测器
    detector = CarDetector(
        distance_threshold=0.3,  # 距离跳变阈值
        min_car_width=0.1,  # 小车最小宽度
        max_car_width=1.0  # 小车最大宽度
    )

    # 数据预处理：按角度排序
    sorted_indices = np.argsort(angles)
    angles_sorted = angles[sorted_indices]
    ranges_sorted = ranges[sorted_indices]

    # 检测小车
    detected_objects = detector.detect_objects(angles_sorted, ranges_sorted)
    target = detector.get_target_position(detected_objects)

    # 输出检测结果
    print("\n=== 小车检测结果 ===")

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
        print(f"\n总共检测到 {len(detected_objects)} 个物体:")
        for i, obj in enumerate(detected_objects):
            print(f"  物体 {i + 1}: 距离={obj['distance']:.2f}m, "
                  f"角度={math.degrees(obj['center_angle']):.1f}°, "
                  f"宽度={obj['spatial_width']:.2f}m")

    # 可视化结果
    visualize_detection(angles_sorted, ranges_sorted, detected_objects)

    return detected_objects, target


def visualize_detection(angles, ranges, detected_objects):
    """
    可视化检测结果

    Args:
        angles: 角度数组
        ranges: 距离数组
        detected_objects: 检测到的物体列表
    """
    plt.figure(figsize=(12, 10))

    # 转换为笛卡尔坐标
    x = [r * math.cos(a) for r, a in zip(ranges, angles)]
    y = [r * math.sin(a) for r, a in zip(ranges, angles)]

    # 绘制所有点
    plt.scatter(x, y, c='blue', s=2, alpha=0.6, label='雷达点')

    # 绘制检测到的物体
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i, obj in enumerate(detected_objects):
        color = colors[i % len(colors)]

        # 绘制物体中心
        center_x = obj['distance'] * math.cos(obj['center_angle'])
        center_y = obj['distance'] * math.sin(obj['center_angle'])
        plt.scatter(center_x, center_y, c=color, s=150, marker='x', linewidth=3,
                    label=f'物体 {i + 1} (d={obj["distance"]:.2f}m)')

        # 绘制边界线
        left_x = obj['distance'] * math.cos(obj['left_angle'])
        left_y = obj['distance'] * math.sin(obj['left_angle'])
        right_x = obj['distance'] * math.cos(obj['right_angle'])
        right_y = obj['distance'] * math.sin(obj['right_angle'])

        plt.plot([0, left_x], [0, left_y], '--', color=color, alpha=0.8, linewidth=2)
        plt.plot([0, right_x], [0, right_y], '--', color=color, alpha=0.8, linewidth=2)

        # 添加物体编号文本
        plt.text(center_x, center_y + 0.1, f'{i + 1}',
                 fontsize=12, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

    # 绘制雷达原点
    plt.scatter(0, 0, c='black', s=100, marker='o', label='雷达位置')

    plt.xlabel('X (米)', fontsize=12)
    plt.ylabel('Y (米)', fontsize=12)
    plt.title('雷达小车检测结果 (从CSV文件)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数 - 从CSV文件检测小车
    """
    # CSV文件路径 - 请根据实际情况修改
    csv_file_path = 'lidar_data/lidar_data0.csv'

    # 处理数据并检测小车
    try:
        detected_objects, target = process_lidar_data(csv_file_path)

        if target:
            print(f"\n=== 最终检测结果 ===")
            print(f"成功检测到目标小车，距离 {target['distance']:.2f} 米，角度 {math.degrees(target['angle']):.1f} 度")
        else:
            print(f"\n=== 最终检测结果 ===")
            print("未能检测到符合条件的目标小车")

    except Exception as e:
        print(f"程序执行出错: {e}")


# 额外功能：批量处理多个CSV文件
def batch_process_csv_files(csv_file_paths):
    """
    批量处理多个CSV文件

    Args:
        csv_file_paths: CSV文件路径列表
    """
    print("=== 批量处理雷达数据 ===")

    results = []

    for i, csv_path in enumerate(csv_file_paths):
        print(f"\n处理文件 {i + 1}/{len(csv_file_paths)}: {csv_path}")

        detected_objects, target = process_lidar_data(csv_path)

        results.append({
            'file_path': csv_path,
            'detected_objects': detected_objects,
            'target': target
        })

    # 汇总结果
    print(f"\n=== 批量处理汇总 ===")
    total_detections = 0
    files_with_targets = 0

    for i, result in enumerate(results):
        obj_count = len(result['detected_objects']) if result['detected_objects'] else 0
        has_target = result['target'] is not None

        total_detections += obj_count
        if has_target:
            files_with_targets += 1

        print(f"文件 {i + 1}: 检测到 {obj_count} 个物体, "
              f"{'有' if has_target else '无'}目标小车")

    print(f"\n总计:")
    print(f"  处理文件数: {len(csv_file_paths)}")
    print(f"  检测到物体总数: {total_detections}")
    print(f"  包含目标小车的文件数: {files_with_targets}")


if __name__ == "__main__":
    # 单文件处理

    main()

    # 如果需要批量处理多个文件，可以取消下面的注释
    # csv_files = [
    #     '/mnt/lidar_data_1.csv',
    #     '/mnt/lidar_data_2.csv',
    #     '/mnt/lidar_data_3.csv'
    # ]
    # batch_process_csv_files(csv_files)