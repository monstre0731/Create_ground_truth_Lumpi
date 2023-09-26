import json
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

## Data path
project_path = '/Users/qingwuliu/Documents/Code/LUMPI_new/LUMPI_new'
data_path = '/Users/qingwuliu/Documents/'
classifyTracks_path = os.path.join(data_path, 'measurement4/classifyTracks.csv')
metadata_path = os.path.join(data_path, 'measurement4/meta.json')
save_path = os.path.join(data_path, 'results/')
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

def draw_points(frame, x, y, image_index):
    color = (0, 255, 0)
    cv2.circle(frame, (int(x), int(y)), 10, color, -1)


def get_transform_parameters(session_ID, data):
    # camera: Intrinsic parameters and extrinsic parameters
    num_session = str(session_ID)  # video 6 for experiment 4
    camera_parameter = data['session'][num_session]
    K = np.array(camera_parameter['intrinsic'])  # intrinsic parameters
    R_t = np.array(camera_parameter['extrinsic'])  # extrinsic parameters

    # Lidar: Transform matrix from lidar coordinate to image coordinate
    tvec = np.array(camera_parameter['tvec'])
    rvec = np.array(camera_parameter['rvec'])
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    return K, R_t, rotation_matrix, tvec.reshape(3)


def points_lid2cam(lidar_point, rotation_matrix, tvec, camera_intrinsic_matrix):
    """
    :param lidar_point: (x, y, z), it will be changed to homogeneous point (x, y, z, 1)
    :param lidar_matrix: 4x4
    :param camera_intrinsic_matrix: camera intrinsic matrix, 3x3
    :return: camera_point_Lid2img: points in image coordinates [x, y, z]
             pixel_point: points in pixel coordinates [u, v]
    """
    K = camera_intrinsic_matrix

    camera_point_Lid2img = np.dot(rotation_matrix, lidar_point) + tvec
    print(f'camera_point_Lid2img: {camera_point_Lid2img}')
    pixel_point = np.dot(K, camera_point_Lid2img)
    print(f'pixel_point: {pixel_point}')

    u = pixel_point[0] / pixel_point[2]
    v = pixel_point[1] / pixel_point[2]
    print(f'u and v: {(u, v)}')
    return camera_point_Lid2img, u, v

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


lidar_points = np.array([-10, 17.279, -0.64])
lidar_points = np.array([43, 10, -1])


K, R_t, rotation_matrix_lidar, tvec = get_transform_parameters(number_session, data)
points_cam_coordinate, u, v = points_lid2cam(lidar_points, rotation_matrix_lidar, tvec, K)
draw_points(frame, u, v, 0)
df = pd.read_csv(classifyTracks_path)
label_dict = {}

for image_index in tqdm(range(1)):
    obj_dict = []
    for index in range(len(df)):
        current_row = df.iloc[index]
        if (0.1 * image_index) <= current_row[0] <= 0.1 * (image_index + 1):
            obj_dict.append(current_row)

    label_dict[f"{image_index}"] = obj_dict

color = (0, 255, 0)  # Green color in BGR format
K, R_t, rotation_matrix_lidar, tvec = get_transform_parameters(number_session, data)

for item in label_dict['0']:
    lidar_point = np.array([item[9], item[10], item[11]])
    points_cam_coordinate, u, v = points_lid2cam(lidar_point, rotation_matrix_lidar, tvec, K)
    draw_points(frame, u, v, 0)
cv2.imwrite(os.path.join(save_path, f'{"%06d" % image_index}.jpg'), frame)
