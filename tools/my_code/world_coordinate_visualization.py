import json
import os
import sys
import numpy as np

####################################################################
# lidar points is just the one in the world coordinate
####################################################################

## project path and add it into sys path ## Data path
project_path = '/Users/qingwuliu/Documents/Code/LUMPI_new/'
data_path = '/Users/qingwuliu/Documents/measurement4/'
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
camera_point_homogeneous = np.append(camera_point, 1) # Homogeneous coordinates
world_point = np.dot(R_t, camera_point_homogeneous)
print(f'world_point: {world_point[:-1]}')

