import numpy as np
import json
import cv2
import numpy as np
import os
import sys

data_path = '/Users/qingwuliu/Documents/'

# Read json
metadata_path = os.path.join(data_path, 'measurement4/meta.json')

with open(metadata_path, 'r') as json_file:
    # Load the JSON data into a Python data structure (usually a dictionary)
    data = json.load(json_file)

number_session = '68'  # video 6 for experiment 4
camera_parameter = data['session'][number_session]
# ['fps', 'tvec', 'rvec', 'extrinsic', 'sessionId', 'type', 'deviceId', 'intrinsic', 'distortion', 'experimentId'])


## World to camera / pixel

# Camera calibration parameters
K = np.array(camera_parameter['intrinsic'])
# Rotation matrix and translation vector
R_t = np.array(camera_parameter['extrinsic'])

# world point
world_point = np.array([0, 0, 0, 1.0])  # 注意最后一个元素是1，表示齐次坐标
# camera_point = np.dot(np.linalg.inv(np.hstack((R, t.reshape(-1, 1)))), world_point)
camera_point = np.dot(R_t, world_point)
# calculate pixel coordinates
pixel_point = np.dot(K, camera_point[:3])  # 仅取前三个元素，忽略齐次坐标

# normalization
u = pixel_point[0] / pixel_point[2]
v = pixel_point[1] / pixel_point[2]
print(f'世界坐标系 -> 像素坐标系')
print("世界坐标系下的点：", world_point[:3])
print("相机坐标系下的点：", camera_point[:3])
print("像素坐标系下的点：", u, v)

## Lidar to camera / pixel
tvec = np.array(camera_parameter['tvec'])
rvec = np.array([camera_parameter['rvec']])
rotation_matrix, _ = cv2.Rodrigues(rvec)
external_matrix = np.hstack((rotation_matrix, tvec.reshape(-1, 1)))
external_matrix = np.vstack((external_matrix, np.array([0, 0, 0, 1])))

lidar_point = np.array([0, 0, 0])
lidar_point_homogeneous = np.hstack((lidar_point, 1))
camera_point_homogeneous = np.dot(external_matrix, lidar_point_homogeneous)
camera_point_Lid2img = camera_point_homogeneous[:3] / camera_point_homogeneous[3]

pixel_point = np.dot(K, camera_point_Lid2img)

u = pixel_point[0] / pixel_point[2]
v = pixel_point[1] / pixel_point[2]
print(f'LiDAR坐标系 -> 像素坐标系')
print("LIDAR坐标系下的点：", lidar_point)
print("相机坐标系下的点：", camera_point_Lid2img)
print("像素坐标系下的点：", u, v)