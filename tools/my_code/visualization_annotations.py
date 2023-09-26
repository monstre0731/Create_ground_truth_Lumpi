import json
import os
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

## project path and add it into sys path
project_path = '/Users/qingwuliu/Documents/Code/LUMPI_new/LUMPI_new'
sys.path.append(project_path)
from utils.utils import draw_points, get_transform_parameters, points_lid2cam


## Data path
data_path = '/Users/qingwuliu/Documents/'
classifyTracks_path = os.path.join(data_path, 'measurement4/classifyTracks.csv')
metadata_path = os.path.join(data_path, 'measurement4/meta.json')
save_path = os.path.join(data_path, 'measurement4/results/')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f'Creating save path, {save_path}')
else:
    print(f'save path has already existed: {save_path}')

## frame rate
fps_lidar = 10
fps_camera = 30

## Read camera parameters from meta.json
with open(metadata_path, 'r') as json_file:
    # Load the JSON data into a Python data structure (usually a dictionary)
    data = json.load(json_file)
def compute_box_3d(obj, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

number_session = 68


df = pd.read_csv(classifyTracks_path)
label_dict = {}

color = (0, 255, 0)  # Green color in BGR format
K, R_t, rotation_matrix_lidar, tvec = get_transform_parameters(number_session, data)

for image_index in tqdm(range(10)):
    print(f'image_index: {image_index}')
    print(f'time duration: {0.1 * image_index} to {0.1* image_index + 1}')
    frame = cv2.imread(os.path.join(data_path, f'measurement4/images/{"%06d" % image_index}.jpg'))
    obj_dict = []
    for index in range(len(df)):
        current_row = df.iloc[index]
        if (0.1 * image_index) <= current_row[0] < 0.1 * (image_index + 1):
            obj_dict.append(current_row)

    label_dict[f"{image_index}"] = obj_dict

    for item in obj_dict:
        lidar_point = np.array([item[9], item[10], item[11]])
        points_cam_coordinate, u, v = points_lid2cam(lidar_point, rotation_matrix_lidar, tvec, K)
        draw_points(frame, u, v, 0)
    cv2.imwrite(os.path.join(save_path, f'{"%06d" % image_index}.jpg'), frame)
    print(f'Please find the image in {save_path}')
