import os
import ydlidar
import time
import math
import sys

sys.path.append('/home/pi/project_demo/lib')
from McLumk_Wheel_Sports import *
import math

speed = 20
    
def turn_right(duration=0.5):
    print("➡️ 开始右转")
    # 控制电机执行右转
    rotate_right(speed)
    time.sleep(duration)
    print("✅ 完成右转")

def turn_left(duration=0.5):
    print("➡️ 开始右转")
    # 控制电机执行右转
    rotate_left(speed)
    time.sleep(duration)
    print("✅ 完成右转")

def reverse(duration=0.8):
    print("🔙 开始倒车")
    # 控制电机倒车
    move_backward(speed)
    time.sleep(duration)
    print("✅ 完成倒车")
    
def check_obstacle(points, safe_distance=0.2):
    front = False
    left = False
    right = False
    too_close = False

    for point in points:
        angle_deg = math.degrees(point.angle) + 90
        distance = point.range

        if 0 < distance < safe_distance:
            # 正前方 ±30°
            if -30 <= angle_deg <= 30:
                front = True
                if distance < 0.3:
                    too_close = True
            # 左侧 30°~150°
            elif 30 < angle_deg <= 150:
                left = True
            # 右侧 -150°~-30°
            elif -150 <= angle_deg < -30:
                right = True

    return {
        "front": front,
        "left": left,
        "right": right,
        "too_close": too_close
    }

ydlidar.os_init()

speed = 100  # Set vehicle speed

ports = ydlidar.lidarPortList()
port = "/dev/ttyUSB0"  # 明确指定你的雷达串口号，避免误选
print(f"使用雷达端口: {port}")

laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)
laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
laser.setlidaropt(ydlidar.LidarPropSampleRate, 3)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
laser.setlidaropt(ydlidar.LidarPropMaxRange, 16.0)
laser.setlidaropt(ydlidar.LidarPropMinRange, 0.08)
laser.setlidaropt(ydlidar.LidarPropIntenstiy, False)

ret = laser.initialize()
if ret:
    ret = laser.turnOn()
    scan = ydlidar.LaserScan()
    last_action = None
    state = "FORWARD"
    
    try:
        while ret and ydlidar.os_isOk():
            r = laser.doProcessSimple(scan)
            if r:
                if scan.config.scan_time > 0:
                    frequency = 1.0 / scan.config.scan_time
                else:
                    frequency = 0.0
    
                print(f"Scan received[ {scan.stamp} ]: {scan.points.size()} points, Frequency: {frequency:.2f} Hz")
    
                obstacle_info = check_obstacle(scan.points)

                if obstacle_info["too_close"]:
                    state = "REVERSE"
                elif obstacle_info["front"]:
                    if obstacle_info["left"] and not obstacle_info["right"]:
                        state = "AVOID_RIGHT"
                    elif obstacle_info["right"] and not obstacle_info["left"]:
                        state = "AVOID_LEFT"
                    else:
                        state = "REVERSE"
                else:
                    state = "FORWARD"

                if state == "AVOID_LEFT" and last_action == "AVOID_RIGHT":
                    print("🔄 切换方向避障")
                    turn_left()
                elif state == "AVOID_RIGHT" and last_action == "AVOID_LEFT":
                    print("🔄 切换方向避障")
                    turn_right()
                elif state == "REVERSE":
                    reverse(duration=0.8)
                else:
                    # 正常执行动作
                    move_forward(speed)
                last_action = state
                
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        print("🛑 用户中断程序")
    
    laser.turnOff()
    laser.disconnecting()
 # 清理资源 Cleaning up resources
del bot
