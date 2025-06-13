# Car
25光电赛

PS：tools\video_ui_qt 和 tools\control_ui_qt文件夹中
..._general.py = ..._ui.py + ..._logic.py

..._general.py是逻辑+ui

..._ui.py是ui

..._logic.py是逻辑

### [video_ui_qt](tools/video_ui_qt)  说明

有两个控件用来显示图像；

左边的默认是原图；右边的可以自由切换(通过右上角的选择控件)

上方预留了接口，加入处理逻辑便可以看到与原图对比查看
```python
# -----------------保留的函数处理接口--------------------
def process_frame(frame):
    processed = frame.copy()
    return processed
```