import sys

sys.path.append('/home/pi/project_demo/lib')
# 导入麦克纳姆轮运动库
from McLumk_Wheel_Sports import *


def move(x, y, angle, move_speed=100, rotate_speed=100):
    try:
        # 处理左右移动
        if x != 0:
            if x < 0:
                move_left(move_speed)
            else:
                move_right(move_speed)
            time.sleep(abs(x) / move_speed)
            stop_robot()

        # 处理前后移动
        if y != 0:
            if y > 0:
                move_forward(move_speed)
            else:
                move_backward(move_speed)
            time.sleep(abs(y) / move_speed)
            stop_robot()

        # 处理原地旋转
        if angle != 0:
            if angle < 0:
                rotate_left(rotate_speed)
            else:
                rotate_right(rotate_speed)
            time.sleep(abs(angle) / rotate_speed)
            stop_robot()

    except KeyboardInterrupt:
        stop_robot()

    finally:
        # 清理资源
        try:
            del bot
        except NameError:
            pass



