import numpy as np

# 定义LiDAR点
lidar_point = np.array([0, 0, 1])

# 定义旋转矩阵A
A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])  # 这是一个单位矩阵，表示没有旋转

# 定义平移矩阵tvec
tvec = np.array([0, 100, -100])

# 计算相机坐标系中的点
camera_point = np.dot(A, lidar_point) + tvec

print("Camera Point:", camera_point)
