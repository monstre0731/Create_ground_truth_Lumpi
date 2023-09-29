import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

####################################################################
# lidar points is just the one in the world coordinate
####################################################################

## project path and add it into sys path ## Data path
project_path = '/Users/qingwuliu/Documents/Code/LUMPI_new/'
data_path = '/Users/qingwuliu/Documents/measurement4/'
save_path = os.path.join(data_path, 'results/')
sys.path.append(project_path)
from utils.utils import get_transform_parameters, points_lid2cam

# camera/lidar info
metadata_path = os.path.join(data_path, 'meta.json')
with open(metadata_path, 'r') as json_file:
    data = json.load(json_file)

##
measurement = str(4)
device = str(6)
num_session = data['device'][device][measurement]

K, R_t, rotation_matrix, tvec = get_transform_parameters(session_ID=num_session, data=data)

lidar_point_1 = [1, 200, 30]
print(f'lidar_point: {lidar_point_1}')
camera_point, _, _ = points_lid2cam(lidar_point_1, rotation_matrix, tvec, K)
camera_point_homogeneous = np.append(camera_point, 1)  # Homogeneous coordinates
world_point = np.dot(R_t, camera_point_homogeneous)
print(f'world_point: {world_point[:-1]}')

### world_coordinate_visualization
csv_file = os.path.join(save_path, 'total_inter.csv')
df = pd.read_csv(csv_file)
total_list = []
for index, row in df.iterrows():
    row_list = row.tolist()
    total_list.append(row_list)
total_array = np.array(total_list)
frame_array = np.unique(total_array[:, 0])
frame_total_number = len(frame_array)

colors = {'0': 'r', '1': 'g', '2': 'b'}

for object_id in [98.0, 1.0, 10.0]:
    print(f'object_id: {object_id}')
    color_index = object_id % 2
    color_row = colors[str(int(color_index))]
    forth_list = []
    for row_item in total_list:
        if row_item[2] == object_id:
            forth_list.append(row_item)
            plt.scatter(row_item[10], row_item[11], color=color_row, marker='x')
# 添加标题和坐标轴标签
plt.title('Scatter Plot Example')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()
