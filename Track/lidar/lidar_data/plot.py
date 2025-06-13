import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 CSV 文件
file_path = 'lidar_data0.csv'
df = pd.read_csv(file_path)

# angles = np.rad2deg(df['angle'])
angles = df['angle']

radii = df['distance']

# 创建极坐标图
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

# 绘图
ax.scatter(angles, radii, s=10)  # s 是点的大小，可调整

# 显示图形
plt.show()
