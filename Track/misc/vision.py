import sys
import time
import threading
import cv2
import math
sys.path.append('/home/pi/project_demo/lib')

# 导入PID控制库（假设您有PID库）
import PID


# 导入舵机控制库（根据您的硬件调整）
# from your_servo_library import servo_control
'''
水平追踪：
    初始化：
        图像
'''
class HorizontalTracker:
    def __init__(self):
        # 死区控制参数
        self.dead_zone = 20  # 像素死区，减少抖动

        # 控制标志
        self.tracking_active = False

        # PID控制器初始化
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

        print(f"水平跟踪系统初始化完成")
        print(f"图像宽度: {self.image_width}像素")
        print(f"图像中心: X={self.image_center_x}像素")
        print(f"舵机中心角度: {self.servo_center_angle}度")

    # ========================开启跟踪=============================
    def set_target_position(self, target_x):
        """
        设置目标在图像中的X坐标位置

        Args:
            target_x (int): 目标的X坐标 (0-640像素)
        """
        # 边界检查
        target_x = max(0, min(self.image_width, target_x))
        # target_x = target_x

        # 误差计算
        error_x = target_x - self.image_center_x

        # 死区控制：只有偏差超过死区才进行控制
        if abs(error_x) > self.dead_zone:
            # PID控制计算
            '''
                目标位置
                图像当前位置
                参数（没看）
            '''
            # ----------------pid参数调节---------------------------
            self.horizontal_pid.SystemOutput = target_x
            self.horizontal_pid.SetStepSignal(self.image_center_x)  # 目标：图像中心
            self.horizontal_pid.SetInertiaTime(0.01, 0.05)

            pid_output = self.horizontal_pid.SystemOutput

            # 将PID输出转换为舵机角度调整量
            angle_adjustment = self.calculate_servo_adjustment(error_x)

            # 更新舵机角度
            new_angle = self.current_servo_angle + angle_adjustment

            # 角度限制
            new_angle = max(self.servo_min_angle, min(self.servo_max_angle, new_angle))

            # 控制舵机
            self.control_servo(new_angle)
            self.current_servo_angle = new_angle

            # 调试信息
            print(f"目标位置: X={target_x}, 偏差: {error_x}, 舵机角度: {new_angle:.1f}°")
        else:
            print(f"目标位置: X={target_x}, 偏差: {error_x} (在死区内，无需调整)")

    # =========调节 像素-> 角度 的映射关系================
    def calculate_servo_adjustment(self, error_x):
        """
        根据水平偏差计算舵机角度调整量
        Args:
            error_x (int): 水平方向偏差（像素）
        Returns:
            float: 舵机角度调整量（度）
        """
        # 转换比例：像素偏差 -> 角度调整
        # 这个比例需要根据您的硬件特性调整
        pixels_per_degree = 8  # 假设每度对应8像素的移动
        # 限制单次调整幅度，避免过大跳跃
        max_adjustment = 10  # 最大单次调整10度

        angle_adjustment = -error_x / pixels_per_degree
        angle_adjustment = max(-max_adjustment, min(max_adjustment, angle_adjustment))

        return angle_adjustment

    def control_servo(self, angle):
        """
        控制舵机转动到指定角度

        Args:
            angle (float): 目标角度
        """
        # 这里需要根据您的硬件接口实现舵机控制
        # 示例代码（需要替换为您的实际硬件控制代码）
        try:
            # 方式1：如果使用类似原代码的舵机控制
            bot.Ctrl_Servo(1, int(angle))

            # 方式2：如果使用其他舵机控制库
            # servo_control.set_angle(channel=1, angle=angle)

            # 方式3：如果使用PWM控制
            # pwm_value = self.angle_to_pwm(angle)
            # GPIO.output(servo_pin, pwm_value)

            print(f"舵机控制: 角度={angle:.1f}°")

        except Exception as e:
            print(f"舵机控制错误: {e}")

    # def angle_to_pwm(self, angle):
    #     """
    #     将角度转换为PWM值（如果使用PWM控制舵机）
    #
    #     Args:
    #         angle (float): 角度 (0-180)
    #
    #     Returns:
    #         int: PWM值
    #     """
    #     # 标准舵机PWM参数
    #     min_pwm = 500  # 0度对应的PWM值
    #     max_pwm = 2500  # 180度对应的PWM值
    #
    #     pwm_value = int(min_pwm + (angle / 180.0) * (max_pwm - min_pwm))
    #     return pwm_value


    # 初始化舵机角度
    def reset_servo(self):
        """复位舵机到中心位置"""
        self.control_servo(self.servo_center_angle)
        self.current_servo_angle = self.servo_center_angle
        print("舵机已复位到中心位置")

    # 设置跟踪状态全局变量
    def start_tracking(self):
        """开始跟踪"""
        self.tracking_active = True
        print("开始水平跟踪")
    def stop_tracking(self):
        """停止跟踪"""
        self.tracking_active = False
        print("停止水平跟踪")

    # 获取每一帧
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

        # 模拟不同的目标位置进行测试
        test_positions = [100, 200, 320, 450, 540]  # X坐标位置

        for target_x in test_positions:
            print(f"\n测试目标位置: X={target_x}")
            tracker.set_target_position(target_x)
            time.sleep(1)  # 等待舵机调整

        # 实时摄像头测试（可选）
        # print("\n开始实时摄像头测试，按'q'退出")
        # while True:
        #     frame = tracker.get_camera_frame()
        #     if frame is not None:
        #         # 这里您可以添加您自己的目标检测算法
        #         # 例如：target_x = your_detection_algorithm(frame)
        #
        #         # 示例：鼠标点击设置目标位置
        #         def mouse_callback(event, x, y, flags, param):
        #             if event == cv2.EVENT_LBUTTONDOWN:
        #                 tracker.set_target_position(x)
        #
        #         cv2.setMouseCallback('Horizontal Tracking', mouse_callback)
        #
        #         # 绘制跟踪信息
        #         display_frame = tracker.draw_tracking_info(frame)
        #         cv2.imshow('Horizontal Tracking', display_frame)
        #
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break

    except KeyboardInterrupt:
        print("\n程序被用户中断")

    finally:
        # 清理资源
        tracker.cleanup()

if __name__ == "__main__":
    main()