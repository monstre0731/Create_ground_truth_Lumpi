import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
# 定义8个点的坐标
points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
          (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

points = [camera_point_list[0],
          camera_point_list[1],
          camera_point_list[2],
          camera_point_list[3],
          camera_point_list[4],
          camera_point_list[5],
          camera_point_list[6],
          camera_point_list[7],]

# 定义连接这些点的面
faces = [[points[0], points[1], points[2], points[3]],
         [points[4], points[5], points[6], points[7]], 
         [points[0], points[1], points[5], points[4]],
         [points[2], points[3], points[7], points[6]],
         [points[0], points[3], points[7], points[4]],
         [points[1], points[2], points[6], points[5]]]

# 创建一个三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制立方体的面
ax.add_collection3d(Poly3DCollection(faces, color='cyan', linewidths=1, edgecolors='r', alpha=0.1))

# 绘制点
x, y, z = zip(*points)
ax.scatter(x, y, z, color='r', s=100)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()


