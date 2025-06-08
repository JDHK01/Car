import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
plt.rcParams['font.family'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class LidarClusterDetector:
    def __init__(self, distance_threshold=0.3, min_car_width=0.1, max_car_width=1.0):
        """
        基于聚类的雷达检测器

        Args:
            distance_threshold: 相邻点距离跳变阈值(米)
            min_car_width: 小车最小宽度(米)
            max_car_width: 小车最大宽度(米)
        """
        self.distance_threshold = distance_threshold
        self.min_car_width = min_car_width
        self.max_car_width = max_car_width
        self.scaler = StandardScaler()

    def polar_to_cartesian(self, ranges, angles):
        """极坐标转笛卡尔坐标"""
        x = [r * math.cos(a) for r, a in zip(ranges, angles)]
        y = [r * math.sin(a) for r, a in zip(ranges, angles)]
        return np.array(x), np.array(y)

    def calculate_distance_between_points(self, r1, angle1, r2, angle2):
        """计算两个极坐标点之间的距离"""
        x1 = r1 * math.cos(angle1)
        y1 = r1 * math.sin(angle1)
        x2 = r2 * math.cos(angle2)
        y2 = r2 * math.sin(angle2)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def segment_points_by_distance(self, angles, ranges):
        """
        基于距离跳变分割点云为不同的segment
        """
        segments = []
        current_segment = [0]  # 当前segment的点索引

        for i in range(len(ranges) - 1):
            # 计算相邻两点的空间距离
            spatial_distance = self.calculate_distance_between_points(
                ranges[i], angles[i], ranges[i + 1], angles[i + 1]
            )

            # 如果距离超过阈值，开始新的segment
            if spatial_distance > self.distance_threshold:
                if len(current_segment) > 1:  # 只保留有多个点的segment
                    segments.append(current_segment)
                current_segment = [i + 1]
            else:
                current_segment.append(i + 1)

        # 添加最后一个segment
        if len(current_segment) > 1:
            segments.append(current_segment)

        return segments

    def extract_segment_features(self, segment_indices, angles, ranges):
        """
        提取segment的特征用于分类

        Features:
        1. 长度 (length)
        2. 平均距离 (avg_distance)
        3. 距离标准差 (distance_std)
        4. 角度跨度 (angular_span)
        5. 点密度 (point_density)
        6. 形状复杂度 (shape_complexity)
        7. 直线拟合误差 (line_fitting_error)
        8. 曲率变化 (curvature_variance)
        """
        if len(segment_indices) < 2:
            return None

        # 获取segment的数据
        seg_angles = [angles[i] for i in segment_indices]
        seg_ranges = [ranges[i] for i in segment_indices]

        # 转换为笛卡尔坐标
        x_coords = [r * math.cos(a) for r, a in zip(seg_ranges, seg_angles)]
        y_coords = [r * math.sin(a) for r, a in zip(seg_ranges, seg_angles)]

        # 特征1: 空间长度（首尾点距离）
        length = math.sqrt((x_coords[-1] - x_coords[0]) ** 2 + (y_coords[-1] - y_coords[0]) ** 2)

        # 特征2: 平均距离
        avg_distance = np.mean(seg_ranges)

        # 特征3: 距离标准差
        distance_std = np.std(seg_ranges)

        # 特征4: 角度跨度
        angular_span = abs(seg_angles[-1] - seg_angles[0])

        # 特征5: 点密度（点数/角度跨度）
        point_density = len(segment_indices) / (angular_span + 1e-6)

        # 特征6: 形状复杂度（相邻点距离变化的标准差）
        adjacent_distances = []
        for i in range(len(x_coords) - 1):
            dist = math.sqrt((x_coords[i + 1] - x_coords[i]) ** 2 + (y_coords[i + 1] - y_coords[i]) ** 2)
            adjacent_distances.append(dist)
        shape_complexity = np.std(adjacent_distances) if adjacent_distances else 0

        # 特征7: 直线拟合误差
        if len(x_coords) >= 2:
            # 使用最小二乘法拟合直线
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            try:
                m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                # 计算点到直线的距离
                line_errors = []
                for x, y in zip(x_coords, y_coords):
                    # 点到直线 y = mx + c 的距离
                    error = abs(m * x - y + c) / math.sqrt(m ** 2 + 1)
                    line_errors.append(error)
                line_fitting_error = np.mean(line_errors)
            except:
                line_fitting_error = 1.0  # 如果拟合失败给个默认值
        else:
            line_fitting_error = 0

        # 特征8: 曲率变化
        curvature_changes = []
        if len(x_coords) >= 3:
            for i in range(1, len(x_coords) - 1):
                # 计算三点构成的角度变化
                v1 = np.array([x_coords[i] - x_coords[i - 1], y_coords[i] - y_coords[i - 1]])
                v2 = np.array([x_coords[i + 1] - x_coords[i], y_coords[i + 1] - y_coords[i]])

                # 避免除零
                norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if norm1 > 1e-6 and norm2 > 1e-6:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1, 1)  # 防止数值误差
                    angle_change = math.acos(cos_angle)
                    curvature_changes.append(angle_change)

        curvature_variance = np.var(curvature_changes) if curvature_changes else 0

        features = {
            'length': length,
            'avg_distance': avg_distance,
            'distance_std': distance_std,
            'angular_span': angular_span,
            'point_density': point_density,
            'shape_complexity': shape_complexity,
            'line_fitting_error': line_fitting_error,
            'curvature_variance': curvature_variance,
            'point_count': len(segment_indices),
            'segment_indices': segment_indices,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'center_x': np.mean(x_coords),
            'center_y': np.mean(y_coords)
        }

        return features

    def classify_segments(self, segment_features):
        """
        使用聚类算法将segments分类为墙壁和小车
        """
        if len(segment_features) < 2:
            return [], []

        # 准备特征矩阵
        feature_matrix = []
        for features in segment_features:
            feature_vector = [
                features['length'],
                features['avg_distance'],
                features['distance_std'],
                features['angular_span'],
                features['point_density'],
                features['shape_complexity'],
                features['line_fitting_error'],
                features['curvature_variance']
            ]
            feature_matrix.append(feature_vector)

        feature_matrix = np.array(feature_matrix)

        # 标准化特征
        feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)

        # 方法1: 基于规则的初步分类
        wall_candidates = []
        car_candidates = []

        for i, features in enumerate(segment_features):
            # 墙壁通常的特征：
            # - 长度较长
            # - 直线拟合误差较小
            # - 点密度较高
            # - 距离标准差较小
            is_wall = (
                    features['length'] > 1.0 and  # 长度大于1米
                    features['line_fitting_error'] < 0.1 and  # 直线拟合误差小
                    features['point_density'] > 10 and  # 点密度高
                    features['distance_std'] < 0.2  # 距离变化小
            )

            # 小车通常的特征：
            # - 长度适中
            # - 角度跨度较小
            # - 距离相对较近
            is_car = (
                    self.min_car_width <= features['length'] <= self.max_car_width and  # 尺寸合适
                    features['angular_span'] < math.pi / 3 and  # 角度跨度不大
                    features['avg_distance'] < 5.0 and  # 距离不太远
                    features['point_count'] >= 3  # 至少3个点
            )

            if is_wall:
                wall_candidates.append(i)
            elif is_car:
                car_candidates.append(i)

        # 方法2: 如果规则分类效果不好，使用K-means聚类
        if len(wall_candidates) == 0 and len(car_candidates) == 0:
            # 使用K-means聚类分为2类
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix_scaled)

            # 根据特征判断哪个类别是墙壁，哪个是小车
            cluster_0_features = [segment_features[i] for i in range(len(segment_features)) if cluster_labels[i] == 0]
            cluster_1_features = [segment_features[i] for i in range(len(segment_features)) if cluster_labels[i] == 1]

            # 墙壁类别通常有更大的平均长度和更小的拟合误差
            avg_length_0 = np.mean([f['length'] for f in cluster_0_features]) if cluster_0_features else 0
            avg_length_1 = np.mean([f['length'] for f in cluster_1_features]) if cluster_1_features else 0

            avg_error_0 = np.mean([f['line_fitting_error'] for f in cluster_0_features]) if cluster_0_features else 1
            avg_error_1 = np.mean([f['line_fitting_error'] for f in cluster_1_features]) if cluster_1_features else 1

            # 判断哪个cluster更像墙壁
            if avg_length_0 > avg_length_1 and avg_error_0 < avg_error_1:
                wall_cluster, car_cluster = 0, 1
            else:
                wall_cluster, car_cluster = 1, 0

            wall_candidates = [i for i in range(len(segment_features)) if cluster_labels[i] == wall_cluster]
            car_candidates = [i for i in range(len(segment_features)) if cluster_labels[i] == car_cluster]

        return wall_candidates, car_candidates

    def detect_objects(self, angles, ranges):
        """
        主检测函数
        """
        # 1. 分割点云
        segments = self.segment_points_by_distance(angles, ranges)

        # 2. 提取每个segment的特征
        segment_features = []
        for segment_indices in segments:
            features = self.extract_segment_features(segment_indices, angles, ranges)
            if features is not None:
                segment_features.append(features)

        # 3. 分类segments
        wall_indices, car_indices = self.classify_segments(segment_features)

        # 4. 整理结果
        walls = [segment_features[i] for i in wall_indices]
        cars = [segment_features[i] for i in car_indices]

        return {
            'walls': walls,
            'cars': cars,
            'all_segments': segment_features
        }

    def get_target_car(self, cars):
        """
        从检测到的小车中选择目标（最近的）
        """
        if not cars:
            return None

        # 选择距离最近的小车
        target = min(cars, key=lambda car: car['avg_distance'])

        # 计算中心角度
        center_angle = math.atan2(target['center_y'], target['center_x'])

        return {
            'distance': target['avg_distance'],
            'angle': center_angle,
            'width': target['length'],
            'center_x': target['center_x'],
            'center_y': target['center_y'],
            'features': target
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


def process_lidar_data_with_clustering(csv_file_path):
    """
    使用聚类方法处理雷达数据并检测墙壁和小车

    Args:
        csv_file_path: CSV文件路径
    """
    # 加载数据
    angles, ranges = load_lidar_data(csv_file_path)

    if angles is None or ranges is None:
        print("数据加载失败，程序退出")
        return None, None

    # 初始化聚类检测器
    detector = LidarClusterDetector(
        distance_threshold=0.3,
        min_car_width=0.1,
        max_car_width=1.0
    )

    # 数据预处理：按角度排序
    sorted_indices = np.argsort(angles)
    angles_sorted = angles[sorted_indices]
    ranges_sorted = ranges[sorted_indices]

    # 检测分类
    results = detector.detect_objects(angles_sorted, ranges_sorted)
    target_car = detector.get_target_car(results['cars'])

    # 输出检测结果
    print("\n=== 基于聚类的检测结果 ===")
    print(f"检测到墙壁段: {len(results['walls'])} 个")
    print(f"检测到小车: {len(results['cars'])} 个")
    print(f"总segment数: {len(results['all_segments'])} 个")

    # 显示墙壁信息
    if results['walls']:
        print(f"\n--- 墙壁详情 ---")
        for i, wall in enumerate(results['walls']):
            print(f"  墙壁 {i + 1}:")
            print(f"    长度: {wall['length']:.2f} 米")
            print(f"    平均距离: {wall['avg_distance']:.2f} 米")
            print(f"    直线拟合误差: {wall['line_fitting_error']:.3f}")
            print(f"    点数: {wall['point_count']}")
            print(f"    中心位置: ({wall['center_x']:.2f}, {wall['center_y']:.2f})")

    # 显示小车信息
    if results['cars']:
        print(f"\n--- 小车详情 ---")
        for i, car in enumerate(results['cars']):
            print(f"  小车 {i + 1}:")
            print(f"    长度: {car['length']:.2f} 米")
            print(f"    平均距离: {car['avg_distance']:.2f} 米")
            print(f"    角度跨度: {math.degrees(car['angular_span']):.1f} 度")
            print(f"    点数: {car['point_count']}")
            print(f"    中心位置: ({car['center_x']:.2f}, {car['center_y']:.2f})")

    # 显示目标小车
    if target_car:
        print(f"\n--- 目标小车 ---")
        print(f"  距离: {target_car['distance']:.2f} 米")
        print(f"  角度: {math.degrees(target_car['angle']):.1f} 度")
        print(f"  宽度: {target_car['width']:.2f} 米")
        print(f"  位置: ({target_car['center_x']:.2f}, {target_car['center_y']:.2f})")
    else:
        print(f"\n--- 目标小车 ---")
        print("  未检测到符合条件的目标小车")

    # 可视化结果
    visualize_cluster_detection(angles_sorted, ranges_sorted, results, target_car)

    return results, target_car


def visualize_cluster_detection(angles, ranges, results, target_car=None):
    """可视化聚类检测结果"""
    plt.figure(figsize=(15, 12))

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 子图1: 极坐标显示
    ax1 = plt.subplot(121, projection='polar')

    # 绘制所有原始点
    ax1.scatter(angles, ranges, c='lightgray', s=1, alpha=0.5, label='原始点')

    # 绘制墙壁
    colors_wall = ['blue', 'navy', 'steelblue', 'dodgerblue']
    for i, wall in enumerate(results['walls']):
        wall_angles = [angles[idx] for idx in wall['segment_indices']]
        wall_ranges = [ranges[idx] for idx in wall['segment_indices']]
        color = colors_wall[i % len(colors_wall)]
        ax1.scatter(wall_angles, wall_ranges, c=color, s=20, alpha=0.8,
                    label=f'墙壁 {i + 1}' if i < 3 else "")

    # 绘制小车
    colors_car = ['red', 'darkred', 'orange', 'purple']
    for i, car in enumerate(results['cars']):
        car_angles = [angles[idx] for idx in car['segment_indices']]
        car_ranges = [ranges[idx] for idx in car['segment_indices']]
        color = colors_car[i % len(colors_car)]
        ax1.scatter(car_angles, car_ranges, c=color, s=30, alpha=0.8,
                    label=f'小车 {i + 1}' if i < 3 else "")

    # 标记目标小车
    if target_car:
        target_angle = target_car['angle']
        target_distance = target_car['distance']
        ax1.scatter(target_angle, target_distance, c='gold', s=200, marker='*',
                    edgecolors='black', linewidth=2, label='目标小车')

    ax1.set_title('极坐标视图', fontsize=14)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True)

    # 子图2: 笛卡尔坐标显示
    ax2 = plt.subplot(122)

    # 转换为笛卡尔坐标
    x = [r * math.cos(a) for r, a in zip(ranges, angles)]
    y = [r * math.sin(a) for r, a in zip(ranges, angles)]

    # 绘制所有原始点
    ax2.scatter(x, y, c='lightgray', s=1, alpha=0.5, label='原始点')

    # 绘制墙壁
    for i, wall in enumerate(results['walls']):
        color = colors_wall[i % len(colors_wall)]
        ax2.scatter(wall['x_coords'], wall['y_coords'], c=color, s=20, alpha=0.8,
                    label=f'墙壁 {i + 1}' if i < 3 else "")
        # 绘制中心点
        ax2.scatter(wall['center_x'], wall['center_y'], c=color, s=100, marker='s',
                    alpha=0.8, edgecolors='black')
        # 添加标签
        ax2.text(wall['center_x'], wall['center_y'] + 0.1, f'W{i + 1}',
                 fontsize=10, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

    # 绘制小车
    for i, car in enumerate(results['cars']):
        color = colors_car[i % len(colors_car)]
        ax2.scatter(car['x_coords'], car['y_coords'], c=color, s=30, alpha=0.8,
                    label=f'小车 {i + 1}' if i < 3 else "")
        # 绘制中心点
        ax2.scatter(car['center_x'], car['center_y'], c=color, s=150, marker='*',
                    alpha=0.8, edgecolors='black')
        # 添加标签
        ax2.text(car['center_x'], car['center_y'] + 0.1, f'C{i + 1}',
                 fontsize=10, ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

    # 标记目标小车
    if target_car:
        ax2.scatter(target_car['center_x'], target_car['center_y'], c='gold', s=250,
                    marker='*', edgecolors='black', linewidth=3, label='目标小车')
        ax2.text(target_car['center_x'], target_car['center_y'] + 0.2, 'TARGET',
                 fontsize=12, ha='center', va='bottom', weight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.8))

    # 绘制雷达位置
    ax2.scatter(0, 0, c='black', s=100, marker='o', label='雷达')

    ax2.set_xlabel('X (米)', fontsize=12)
    ax2.set_ylabel('Y (米)', fontsize=12)
    ax2.set_title('笛卡尔坐标视图', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()
    plt.show()


def analyze_segment_features(results):
    """分析segment特征的分布"""
    if not results['all_segments']:
        print("没有检测到任何segment")
        return

    print(f"\n=== Segment特征分析 ===")
    print(f"总segment数: {len(results['all_segments'])}")

    # 统计特征
    lengths = [seg['length'] for seg in results['all_segments']]
    distances = [seg['avg_distance'] for seg in results['all_segments']]
    errors = [seg['line_fitting_error'] for seg in results['all_segments']]
    point_counts = [seg['point_count'] for seg in results['all_segments']]

    print(f"\n特征统计:")
    print(
        f"  长度: 平均={np.mean(lengths):.2f}, 标准差={np.std(lengths):.2f}, 范围=[{np.min(lengths):.2f}, {np.max(lengths):.2f}]")
    print(
        f"  距离: 平均={np.mean(distances):.2f}, 标准差={np.std(distances):.2f}, 范围=[{np.min(distances):.2f}, {np.max(distances):.2f}]")
    print(
        f"  拟合误差: 平均={np.mean(errors):.3f}, 标准差={np.std(errors):.3f}, 范围=[{np.min(errors):.3f}, {np.max(errors):.3f}]")
    print(
        f"  点数: 平均={np.mean(point_counts):.1f}, 标准差={np.std(point_counts):.1f}, 范围=[{np.min(point_counts)}, {np.max(point_counts)}]")


def test_with_simulated_data():
    """使用模拟数据测试聚类检测"""
    print("=== 生成模拟数据 ===")

    # 创建模拟数据：墙壁 + 小车
    angles = np.linspace(-math.pi, math.pi, 360)
    ranges = []

    for angle in angles:
        # 模拟环境：四面墙 + 一个小车
        if -math.pi / 4 <= angle <= math.pi / 4:  # 前方
            if -0.2 <= angle <= 0.2:  # 小车在前方
                ranges.append(2.0)
            else:  # 前墙
                ranges.append(4.0)
        elif math.pi / 4 < angle <= 3 * math.pi / 4:  # 右墙
            ranges.append(3.0)
        elif -3 * math.pi / 4 <= angle < -math.pi / 4:  # 左墙
            ranges.append(3.0)
        else:  # 后墙
            ranges.append(4.0)

    # 添加噪声
    ranges = [r + np.random.normal(0, 0.05) for r in ranges]

    # 检测
    detector = LidarClusterDetector()
    results = detector.detect_objects(angles, ranges)
    target_car = detector.get_target_car(results['cars'])

    print("=== 模拟测试结果 ===")
    print(f"检测到墙壁: {len(results['walls'])} 个")
    print(f"检测到小车: {len(results['cars'])} 个")

    # 分析特征
    analyze_segment_features(results)

    # 可视化
    visualize_cluster_detection(angles, ranges, results, target_car)

    return results, target_car


def main():
    """
    主函数 - 从CSV文件进行基于聚类的检测
    """
    # CSV文件路径 - 请根据实际情况修改
    csv_file_path = '/mnt/lidar_data.csv'

    try:
        print("=== 基于聚类的雷达数据分析 ===")
        results, target_car = process_lidar_data_with_clustering(csv_file_path)

        if results:
            # 分析segment特征
            analyze_segment_features(results)

            # 输出最终检测结果
            print(f"\n=== 最终检测结果 ===")
            if target_car:
                print(f"成功检测到目标小车:")
                print(f"  距离: {target_car['distance']:.2f} 米")
                print(f"  角度: {math.degrees(target_car['angle']):.1f} 度")
                print(f"  宽度: {target_car['width']:.2f} 米")
                print(f"  位置: ({target_car['center_x']:.2f}, {target_car['center_y']:.2f})")

                # 检查是否符合跟随条件
                distance = target_car['distance']
                angle_deg = abs(math.degrees(target_car['angle']))

                print(f"\n=== 跟随状态评估 ===")
                if distance < 0.5:
                    print("⚠️  警告: 目标距离过近，建议减速或停止")
                elif distance > 5.0:
                    print("⚠️  警告: 目标距离过远，可能跟丢")
                elif angle_deg > 30:
                    print("⚠️  警告: 目标偏角过大，需要调整方向")
                else:
                    print("✅ 目标状态良好，可以正常跟随")

                # 给出控制建议
                print(f"\n=== 控制建议 ===")
                if angle_deg > 5:
                    turn_direction = "左" if target_car['angle'] > 0 else "右"
                    print(f"建议向{turn_direction}转 {angle_deg:.1f} 度")
                else:
                    print("方向保持良好")

                if distance > 2.0:
                    print("建议加速前进")
                elif distance < 1.0:
                    print("建议减速")
                else:
                    print("速度保持适中")
            else:
                print("❌ 未检测到目标小车")
                print("建议:")
                print("  1. 检查周围环境")
                print("  2. 调整雷达参数")
                print("  3. 可能需要重新搜索目标")

            # 输出检测统计
            print(f"\n=== 检测统计 ===")
            print(f"墙壁数量: {len(results['walls'])}")
            print(f"小车数量: {len(results['cars'])}")
            print(f"总segment数: {len(results['all_segments'])}")

            return results, target_car
        else:
            print("❌ 数据处理失败")
            return None, None

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {csv_file_path}")
        print("请检查文件路径是否正确")

        # 提供测试选项
        print("\n是否使用模拟数据进行测试? (y/n)")
        user_input = input().strip().lower()
        if user_input in ['y', 'yes']:
            print("\n=== 切换到模拟数据测试 ===")
            return test_with_simulated_data()
        else:
            return None, None

    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        print("错误详情:")
        import traceback
        traceback.print_exc()
        return None, None


def interactive_test():
    """
    交互式测试函数，允许用户选择不同的测试模式
    """
    print("=== 雷达聚类检测系统 ===")
    print("请选择测试模式:")
    print("1. 从CSV文件读取数据")
    print("2. 使用模拟数据测试")
    print("3. 退出")

    while True:
        choice = input("\n请输入选项 (1-3): ").strip()

        if choice == '1':
            csv_path = input("请输入CSV文件路径 (直接回车使用默认路径): ").strip()
            if not csv_path:
                csv_path = 'lidar_data2.csv'

            # 临时修改路径并运行
            global csv_file_path
            original_path = csv_file_path if 'csv_file_path' in globals() else None
            try:
                # 修改main函数中的路径
                results, target_car = process_lidar_data_with_clustering(csv_path)
                if results:
                    analyze_segment_features(results)
                return results, target_car
            except Exception as e:
                print(f"处理CSV文件时出错: {e}")
                continue

        elif choice == '2':
            print("\n=== 开始模拟数据测试 ===")
            return test_with_simulated_data()

        elif choice == '3':
            print("程序退出")
            return None, None

        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    # 可以选择运行方式
    import sys

    if len(sys.argv) > 1:
        # 如果有命令行参数，直接运行main
        if sys.argv[1] == '--simulate':
            # 运行模拟测试
            test_with_simulated_data()
        elif sys.argv[1] == '--file' and len(sys.argv) > 2:
            # 指定文件路径
            csv_path = sys.argv[2]
            process_lidar_data_with_clustering(csv_path)
        else:
            # 默认运行main
            main()
    else:
        # 交互式模式
        interactive_test()