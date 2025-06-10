import sys
import time
import threading
import cv2
import math
from collections import deque

sys.path.append('/home/pi/project_demo/lib')

import yolo

# 导入PID控制库
import PID

# 导入舵机控制库（根据您的硬件调整）
from McLumk_Wheel_Sports import *


class MovementRecord:
    """运动记录类，用于记录和还原小车运动"""

    def __init__(self):
        self.movements = deque(maxlen=100)  # 最多记录100个动作

    def record_rotation(self, direction, speed, duration):
        """记录旋转动作"""
        self.movements.append({
            'type': 'rotation',
            'direction': direction,  # 'left' or 'right'
            'speed': speed,
            'duration': duration,
            'timestamp': time.time()
        })

    def record_forward(self, speed, duration):
        """记录前进动作"""
        self.movements.append({
            'type': 'forward',
            'speed': speed,
            'duration': duration,
            'timestamp': time.time()
        })

    def get_reverse_movements(self):
        """获取反向运动序列"""
        reverse_moves = []
        for move in reversed(self.movements):
            if move['type'] == 'rotation':
                # 旋转反向
                reverse_dir = 'right' if move['direction'] == 'left' else 'left'
                reverse_moves.append({
                    'type': 'rotation',
                    'direction': reverse_dir,
                    'speed': move['speed'],
                    'duration': move['duration']
                })
            elif move['type'] == 'forward':
                # 前进变后退
                reverse_moves.append({
                    'type': 'backward',
                    'speed': move['speed'],
                    'duration': move['duration']
                })
        return reverse_moves

    def clear(self):
        """清空运动记录"""
        self.movements.clear()


class EnhancedTracker:
    def __init__(self):
        # =========参数调节设置================
        '''
            死区宽度设置；
            是否跟踪到的标识位
            PID参数
            映射数值：作用类似于PID参数的P
            小车运动的参数调节
        '''
        # 死区控制参数
        self.dead_zone_horizontal = 20  # 水平像素死区
        self.dead_zone_vertical = 15  # 垂直像素死区
        # 控制标志
        self.tracking_active = False
        # PID控制器初始化
        self.horizontal_pid = PID.PositionalPID(0.8, 0, 0.2)
        self.vertical_pid = PID.PositionalPID(0.6, 0, 0.15)  # 垂直方向稍微温和一些
        # PID输出到舵机角度的映射值
        self.horizontal_pid_scale = 0.5
        self.vertical_pid_scale = 0.03  # 垂直方向更精细
        # 小车运动参数
        self.robot_rotation_threshold = 45  # 舵机角度超过此值时开始旋转小车
        self.robot_speed = 30  # 小车运动速度
        self.rotation_time_per_degree = 0.02  # 每度旋转需要的时间（需要根据实际调试）

        # 摄像头初始化
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 图像参数
        self.image_width = 640
        self.image_height = 480
        self.image_center_x = self.image_width // 2  # 320像素
        self.image_center_y = self.image_height // 2  # 240像素

        # 水平舵机参数（舵机1）
        self.horizontal_servo_center = 90
        self.horizontal_servo_min = 0
        self.horizontal_servo_max = 180
        self.current_horizontal_angle = self.horizontal_servo_center

        # 垂直舵机参数（舵机2）
        self.vertical_servo_center = 0
        self.vertical_servo_min = 0  # 限制垂直范围，避免撞到地面
        self.vertical_servo_max = 90
        self.current_vertical_angle = self.vertical_servo_center

        # 运动记录
        self.movement_recorder = MovementRecord()

        # 小车运动锁，防止同时执行多个运动指令
        self.movement_lock = threading.Lock()

        print(f"增强跟踪系统初始化完成")
        print(f"图像尺寸: {self.image_width}x{self.image_height}")
        print(f"图像中心: X={self.image_center_x}, Y={self.image_center_y}")
        print(f"水平舵机中心: {self.horizontal_servo_center}°")
        print(f"垂直舵机中心: {self.vertical_servo_center}°")

    def set_target_position(self, target_x, target_y):
        """
        设置目标在图像中的位置并进行跟踪控制

        Args:
            target_x (int): 目标的X坐标 (0-640像素)
            target_y (int): 目标的Y坐标 (0-480像素)
        """
        # 边界检查
        target_x = max(0, min(self.image_width, target_x))
        target_y = max(0, min(self.image_height, target_y))

        # 计算误差
        error_x = target_x - self.image_center_x
        error_y = target_y - self.image_center_y

        print(f"目标位置: X={target_x}, Y={target_y}, 偏差: X={error_x}, Y={error_y}")

        # 水平方向控制
        self._control_horizontal(target_x, error_x)
        # 垂直方向控制
        self._control_vertical(target_y, error_y)
        # 检查是否需要旋转小车
        self._check_robot_rotation()

    def _control_horizontal(self, target_x, error_x):
        """水平方向控制"""
        if abs(error_x) > self.dead_zone_horizontal:
            # PID控制计算
            self.horizontal_pid.SystemOutput = target_x
            self.horizontal_pid.SetStepSignal(self.image_center_x)
            self.horizontal_pid.SetInertiaTime(0.01, 0.05)

            # 计算角度调整量
            angle_adjustment = -error_x * self.horizontal_pid_scale

            # 限制单次调整幅度
            max_adjustment = 8
            angle_adjustment = max(-max_adjustment, min(max_adjustment, angle_adjustment))

            # 更新舵机角度
            new_angle = self.current_horizontal_angle + angle_adjustment
            new_angle = max(self.horizontal_servo_min, min(self.horizontal_servo_max, new_angle))

            # 控制舵机
            self.control_servo(1, new_angle)
            self.current_horizontal_angle = new_angle

            print(f"水平调整: 角度调整={angle_adjustment:.1f}°, 当前角度={new_angle:.1f}°")

    def _control_vertical(self, target_y, error_y):
        """垂直方向控制"""
        if abs(error_y) > self.dead_zone_vertical:
            # PID控制计算
            self.vertical_pid.SystemOutput = target_y
            self.vertical_pid.SetStepSignal(self.image_center_y)
            self.vertical_pid.SetInertiaTime(0.01, 0.05)

            # 计算角度调整量（注意：Y坐标向下增加，但舵机向上应该角度减小）
            angle_adjustment = error_y * self.vertical_pid_scale  # 不加负号，因为Y轴方向

            # 限制单次调整幅度
            max_adjustment = 6
            angle_adjustment = max(-max_adjustment, min(max_adjustment, angle_adjustment))

            # 更新舵机角度
            new_angle = self.current_vertical_angle + angle_adjustment
            new_angle = max(self.vertical_servo_min, min(self.vertical_servo_max, new_angle))

            # 控制舵机
            self.control_servo(2, new_angle)
            self.current_vertical_angle = new_angle

            print(f"垂直调整: 角度调整={angle_adjustment:.1f}°, 当前角度={new_angle:.1f}°")

    def _check_robot_rotation(self):
        """检查是否需要旋转小车来辅助跟踪"""
        # 检查水平舵机是否接近极限位置
        horizontal_deviation = abs(self.current_horizontal_angle - self.horizontal_servo_center)

        if horizontal_deviation > self.robot_rotation_threshold:
            # 需要旋转小车
            if self.current_horizontal_angle > self.horizontal_servo_center:
                # 目标在右侧，小车右转
                self._rotate_robot_with_advance('right', horizontal_deviation)
            else:
                # 目标在左侧，小车左转
                self._rotate_robot_with_advance('left', horizontal_deviation)

    def _rotate_robot_with_advance(self, direction, angle_deviation):
        """旋转小车并前进，同时记录运动"""
        with self.movement_lock:
            try:
                # 计算旋转时间和前进时间
                rotation_angle = min(30, angle_deviation - self.robot_rotation_threshold)  # 最多旋转30度
                rotation_time = rotation_angle * self.rotation_time_per_degree
                advance_time = rotation_time * 0.5  # 前进时间为旋转时间的一半

                print(f"执行小车运动: {direction}转 {rotation_angle:.1f}度, 同时前进")

                # 同时开始旋转和前进
                if direction == 'left':
                    rotate_left(self.robot_speed)
                else:
                    rotate_right(self.robot_speed)

                # 同时前进一小段距离
                move_forward(self.robot_speed)

                # 等待旋转和前进完成
                time.sleep(rotation_time)
                stop_robot()  # 同时停止旋转和前进

                # 记录运动
                self.movement_recorder.record_rotation(direction, self.robot_speed, rotation_time)
                self.movement_recorder.record_forward(self.robot_speed, advance_time)

                # 旋转后将舵机角度调整回中心附近
                center_adjustment = (self.current_horizontal_angle - self.horizontal_servo_center) * 0.3
                new_horizontal_angle = self.horizontal_servo_center + center_adjustment
                new_horizontal_angle = max(self.horizontal_servo_min,
                                           min(self.horizontal_servo_max, new_horizontal_angle))

                self.control_servo(1, new_horizontal_angle)
                self.current_horizontal_angle = new_horizontal_angle

                print(f"小车运动完成，舵机调整到 {new_horizontal_angle:.1f}°")

            except Exception as e:
                print(f"小车运动控制错误: {e}")
                stop_robot()

    def _advance_briefly(self, duration):
        """短暂前进"""
        try:
            time.sleep(0.1)  # 稍微延迟，让旋转先开始
            move_forward(self.robot_speed)  # 使用正确的前进函数
            time.sleep(duration)
            # 停止前进在主线程的stop_robot()中统一处理
        except Exception as e:
            print(f"前进控制错误: {e}")
            stop_robot()

    def control_servo(self, servo_id, angle):
        """
        控制舵机转动到指定角度

        Args:
            servo_id (int): 舵机ID (1=水平, 2=垂直)
            angle (float): 目标角度
        """
        try:
            bot.Ctrl_Servo(servo_id, int(angle))
            servo_name = "水平" if servo_id == 1 else "垂直"
            print(f"{servo_name}舵机控制: 角度={angle:.1f}°")

        except Exception as e:
            print(f"舵机{servo_id}控制错误: {e}")

    def reset_servos(self):
        """复位所有舵机到中心位置"""
        self.control_servo(1, self.horizontal_servo_center)
        self.control_servo(2, self.vertical_servo_center)
        self.current_horizontal_angle = self.horizontal_servo_center
        self.current_vertical_angle = self.vertical_servo_center
        print("所有舵机已复位到中心位置")

    def return_to_origin(self):
        """根据运动记录返回原点"""
        print("开始返回原点...")
        reverse_moves = self.movement_recorder.get_reverse_movements()

        with self.movement_lock:
            for move in reverse_moves:
                try:
                    if move['type'] == 'rotation':
                        print(f"反向旋转: {move['direction']}, 时长: {move['duration']:.2f}s")
                        if move['direction'] == 'left':
                            rotate_left(move['speed'])
                        else:
                            rotate_right(move['speed'])
                        time.sleep(move['duration'])
                        stop_robot()

                    elif move['type'] == 'backward':
                        print(f"后退, 时长: {move['duration']:.2f}s")
                        move_backward(move['speed'])  # 使用正确的后退函数
                        time.sleep(move['duration'])
                        stop_robot()

                    time.sleep(0.2)  # 每个动作之间稍作停顿

                except Exception as e:
                    print(f"返回原点时发生错误: {e}")
                    stop_robot()

        # 清空运动记录
        self.movement_recorder.clear()
        print("返回原点完成")

    def start_tracking(self):
        """开始跟踪"""
        self.tracking_active = True
        print("开始增强跟踪模式")

    def stop_tracking(self):
        """停止跟踪"""
        self.tracking_active = False
        stop_robot()  # 确保小车停止
        print("停止增强跟踪模式")

    def get_camera_frame(self):
        """获取摄像头画面"""
        ret, frame = self.camera.read()
        if ret:
            return frame
        else:
            print("没有检测到内容")
            return None

    def cleanup(self):
        """清理资源"""
        self.stop_tracking()
        self.reset_servos()
        if self.camera.isOpened():
            self.camera.release()
        print("资源清理完成")


# 使用示例
def main():
    # 创建增强跟踪器
    tracker = EnhancedTracker()
    # 创建识别器
    yolo_detector = yolo.YOLOv5Detector(model_path='model/yolov5s_bs1.om', device_id=0)

    try:
        # 复位舵机
        tracker.reset_servos()

        time.sleep(1)

        # 开始跟踪
        tracker.start_tracking()

        frame_count = 0
        while tracker.tracking_active:
            frame = tracker.get_camera_frame()
            if frame is not None:
                frame_count += 1

                # 使用YOLO检测目标，这里需要修改detect_frame方法返回x,y坐标
                detection_result = yolo_detector.detect_frame(frame, conf_thres=0.4, iou_thres=0.5)

                if detection_result is not None:
                    # 假设detection_result返回(target_x, target_y)
                    # 如果原来只返回x坐标，需要修改YOLO检测器或在这里处理
                    if isinstance(detection_result, tuple) and len(detection_result) == 2:
                        target_x, target_y = detection_result
                        tracker.set_target_position(target_x, target_y)
                    else:
                        # 兼容原来只返回x坐标的情况
                        target_x = detection_result
                        target_y = tracker.image_center_y  # 使用图像中心Y坐标
                        tracker.set_target_position(target_x, target_y)
                else:
                    if frame_count % 30 == 0:  # 每30帧提示一次
                        print("未检测到目标")

                time.sleep(0.03)  # 控制帧率

    except KeyboardInterrupt:
        print("\n程序被用户中断，准备返回原点...")

        # 返回原点
        tracker.return_to_origin()

    finally:
        # 清理资源
        tracker.cleanup()


if __name__ == "__main__":
    main()