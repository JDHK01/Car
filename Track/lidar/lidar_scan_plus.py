import os
import ydlidar
import time
import sys
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

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
    """主函数 - 基于聚类的实时检测"""
    # 初始化雷达
    laser = setup_lidar()
    scan = ydlidar.LaserScan()

    # 初始化检测器
    detector = LidarClusterDetector(
        distance_threshold=0.3,
        min_car_width=0.1,
        max_car_width=1.0
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

                if len(angles) > 10:  # 确保有足够的点
                    # 排序数据
                    sorted_data = sorted(zip(angles, ranges))
                    angles, ranges = zip(*sorted_data)
                    angles = list(angles)
                    ranges = list(ranges)

                    # 检测分类
                    results = detector.detect_objects(angles, ranges)
                    target_car = detector.get_target_car(results['cars'])

                    scan_count += 1
                    print(f"\n=== 扫描 #{scan_count} ===")
                    print(f"检测到墙壁段: {len(results['walls'])} 个")
                    print(f"检测到小车: {len(results['cars'])} 个")

                    # 显示墙壁信息
                    for i, wall in enumerate(results['walls']):
                        print(f"  墙壁 {i + 1}: 长度={wall['length']:.2f}m, "
                              f"距离={wall['avg_distance']:.2f}m, "
                              f"拟合误差={wall['line_fitting_error']:.3f}")

                    # 显示小车信息
                    for i, car in enumerate(results['cars']):
                        print(f"  小车 {i + 1}: 长度={car['length']:.2f}m, "
                              f"距离={car['avg_distance']:.2f}m, "
                              f"位置=({car['center_x']:.2f}, {car['center_y']:.2f})")

                    # 显示目标小车
                    if target_car:
                        print(f"\n>>> 目标小车:")
                        print(f"    距离: {target_car['distance']:.2f} 米")
                        print(f"    角度: {math.degrees(target_car['angle']):.1f} 度")
                        print(f"    宽度: {target_car['width']:.2f} 米")
                        print(f"    位置: ({target_car['center_x']:.2f}, {target_car['center_y']:.2f})")
                    else:
                        print("\n>>> 未检测到目标小车")

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n检测停止")
        finally:
            laser.turnOff()

    laser.disconnecting()


def visualize_cluster_detection(angles, ranges, results):
    """可视化聚类检测结果"""
    plt.figure(figsize=(12, 10))

    # 转换为笛卡尔坐标
    x = [r * math.cos(a) for r, a in zip(ranges, angles)]
    y = [r * math.sin(a) for r, a in zip(ranges, angles)]

    # 绘制所有原始点
    plt.scatter(x, y, c='lightgray', s=1, alpha=0.5, label='原始点')

    # 绘制墙壁
    for i, wall in enumerate(results['walls']):
        plt.scatter(wall['x_coords'], wall['y_coords'],
                    c='blue', s=20, alpha=0.8, label=f'墙壁 {i + 1}' if i < 3 else "")
        # 绘制中心点
        plt.scatter(wall['center_x'], wall['center_y'],
                    c='navy', s=100, marker='s', alpha=0.8)

    # 绘制小车
    for i, car in enumerate(results['cars']):
        plt.scatter(car['x_coords'], car['y_coords'],
                    c='red', s=30, alpha=0.8, label=f'小车 {i + 1}' if i < 3 else "")
        # 绘制中心点
        plt.scatter(car['center_x'], car['center_y'],
                    c='darkred', s=150, marker='*', alpha=0.8)

    # 绘制雷达位置
    plt.scatter(0, 0, c='black', s=100, marker='o', label='雷达')

    plt.xlabel('X (米)')
    plt.ylabel('Y (米)')
    plt.title('基于聚类的墙壁和小车检测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


# 测试函数
def test_with_simulated_data():
    """使用模拟数据测试聚类检测"""
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

    print("=== 模拟测试结果 ===")
    print(f"检测到墙壁: {len(results['walls'])} 个")
    print(f"检测到小车: {len(results['cars'])} 个")

    # 可视化
    visualize_cluster_detection(angles, ranges, results)


if __name__ == "__main__":
    # 选择运行模式
    mode = input("选择模式 (1: 实时检测, 2: 模拟测试): ")
    if mode == "2":
        test_with_simulated_data()
    else:
        main()