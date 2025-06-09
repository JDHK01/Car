import sys
import time
import threading
import cv2
import math

sys.path.append('/home/pi/project_demo/lib')

# 导入PID控制库
import PID

# 导入舵机控制库（根据您的硬件调整）
from McLumk_Wheel_Sports import *


class HorizontalTracker:
    def __init__(self):
        # 死区控制参数
        self.dead_zone = 20  # 像素死区，减少抖动

        # 控制标志
        self.tracking_active = False

        # PID控制器初始化 - 参考PDF中的参数
        self.horizontal_pid = PID.PositionalPID(0.8, 0, 0.2)

        # 摄像头初始化
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 图像参数
        self.image_width = 640
        self.image_center_x = self.image_width // 2  # 320像素

        # 舵机参数
        self.servo_center_angle = 90  # 舵机中心角度
        self.servo_min_angle = 0  # 舵机最小角度
        self.servo_max_angle = 180  # 舵机最大角度
        self.current_servo_angle = self.servo_center_angle  # 角度初始化

        # PID输出到舵机角度的映射值
        self.pid_to_servo_scale = 0.1  # PID输出转换为角度的比例

        print(f"水平跟踪系统初始化完成")
        print(f"图像宽度: {self.image_width}像素")
        print(f"图像中心: X={self.image_center_x}像素")
        print(f"舵机中心角度: {self.servo_center_angle}度")

    def set_target_position(self, target_x):
        """
        设置目标在图像中的X坐标位置

        Args:
            target_x (int): 目标的X坐标 (0-640像素)
        """
        # 边界检查
        target_x = max(0, min(self.image_width, target_x))

        # 误差计算
        error_x = target_x - self.image_center_x

        # 死区控制：只有偏差超过死区才进行控制
        if abs(error_x) > self.dead_zone:
            # ===== 修正的PID控制计算 =====
            # 参考PDF中的正确用法：
            # 1. SystemOutput 设置为当前值（目标位置）
            # 2. SetStepSignal 设置为目标值（图像中心）
            self.horizontal_pid.SystemOutput = target_x
            self.horizontal_pid.SetStepSignal(self.image_center_x)
            self.horizontal_pid.SetInertiaTime(0.01, 0.05)

            # 获取PID计算后的输出值
            pid_output = self.horizontal_pid.SystemOutput

            # 计算PID输出与目标位置的差值，这个差值用于控制舵机
            pid_adjustment = pid_output - self.image_center_x

            # 将PID输出转换为舵机角度调整量
            angle_adjustment = -pid_adjustment * self.pid_to_servo_scale

            # 限制单次调整幅度
            max_adjustment = 10  # 最大单次调整10度
            angle_adjustment = max(-max_adjustment, min(max_adjustment, angle_adjustment))

            # 更新舵机角度
            new_angle = self.current_servo_angle + angle_adjustment

            # 角度限制
            new_angle = max(self.servo_min_angle, min(self.servo_max_angle, new_angle))

            # 控制舵机
            self.control_servo(new_angle)
            self.current_servo_angle = new_angle

            # 调试信息
            print(f"目标位置: X={target_x}, 偏差: {error_x}, PID输出: {pid_output:.1f}, 舵机角度: {new_angle:.1f}°")
        else:
            print(f"目标位置: X={target_x}, 偏差: {error_x} (在死区内，无需调整)")

    def control_servo(self, angle):
        """
        控制舵机转动到指定角度

        Args:
            angle (float): 目标角度
        """
        try:
            # 使用与PDF代码相同的舵机控制方式
            bot.Ctrl_Servo(1, int(angle))
            print(f"舵机控制: 角度={angle:.1f}°")

        except Exception as e:
            print(f"舵机控制错误: {e}")

    def reset_servo(self):
        """复位舵机到中心位置"""
        self.control_servo(self.servo_center_angle)
        self.current_servo_angle = self.servo_center_angle
        print("舵机已复位到中心位置")

    def start_tracking(self):
        """开始跟踪"""
        self.tracking_active = True
        print("开始水平跟踪")

    def stop_tracking(self):
        """停止跟踪"""
        self.tracking_active = False
        print("停止水平跟踪")

    def get_camera_frame(self):
        """获取摄像头画面"""
        ret, frame = self.camera.read()
        if ret:
            return frame
        else:
            print("没有检测到内容")
            return None

    def draw_tracking_info(self, frame, target_x=None):
        """
        在图像上绘制跟踪信息

        Args:
            frame: 图像帧
            target_x: 目标X坐标
        """
        # 绘制中心线
        cv2.line(frame, (self.image_center_x, 0),
                 (self.image_center_x, frame.shape[0]), (0, 255, 0), 2)

        # 绘制死区范围
        dead_zone_left = self.image_center_x - self.dead_zone
        dead_zone_right = self.image_center_x + self.dead_zone
        cv2.line(frame, (dead_zone_left, 0),
                 (dead_zone_left, frame.shape[0]), (255, 255, 0), 1)
        cv2.line(frame, (dead_zone_right, 0),
                 (dead_zone_right, frame.shape[0]), (255, 255, 0), 1)

        # 绘制目标位置
        if target_x is not None:
            cv2.circle(frame, (int(target_x), frame.shape[0] // 2), 10, (0, 0, 255), -1)
            error_x = target_x - self.image_center_x
            cv2.putText(frame, f"Target: X={target_x}, Error: {error_x}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 显示舵机角度
        cv2.putText(frame, f"Servo: {self.current_servo_angle:.1f}°",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def cleanup(self):
        """清理资源"""
        self.stop_tracking()
        self.reset_servo()
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()
        print("资源清理完成")

    def track_with_pid(self):
        """
        实时跟踪循环（类似PDF中的Face_Follow函数）
        """
        while self.tracking_active:
            frame = self.get_camera_frame()
            if frame is not None:
                # 这里应该添加您的目标检测算法
                # 例如：target_x = detect_target(frame)

                # 为演示目的，这里使用鼠标点击位置
                def mouse_callback(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self.set_target_position(x)

                cv2.setMouseCallback('Horizontal Tracking', mouse_callback)

                # 绘制跟踪信息
                display_frame = self.draw_tracking_info(frame)
                cv2.imshow('Horizontal Tracking', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


# 使用示例
def main():
    # 创建跟踪器
    tracker = HorizontalTracker()

    try:
        # 复位舵机
        tracker.reset_servo()
        time.sleep(1)

        # 开始跟踪
        tracker.start_tracking()

        # 测试1：固定位置测试
        print("\n===== 固定位置测试 =====")
        test_positions = [100, 200, 320, 450, 540]  # X坐标位置

        for target_x in test_positions:
            print(f"\n测试目标位置: X={target_x}")
            tracker.set_target_position(target_x)
            time.sleep(2)  # 等待舵机调整

        # 测试2：实时跟踪测试
        print("\n===== 开始实时跟踪测试，点击画面设置目标，按'q'退出 =====")
        tracker.track_with_pid()

    except KeyboardInterrupt:
        print("\n程序被用户中断")

    finally:
        # 清理资源
        tracker.cleanup()


if __name__ == "__main__":
    main()