import json
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

## Data path
project_path = '/Users/qingwuliu/Documents/Code/LUMPI_new'
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
    lidar_matrix = np.hstack((rotation_matrix, tvec.reshape(-1, 1)))
    lidar_matrix = np.vstack((lidar_matrix, np.array([0, 0, 0, 1])))

    return K, R_t, lidar_matrix


def points_lid2cam(lidar_point, lidar_matrix, camera_intrinsic_matrix):
    """
    :param lidar_point: (x, y, z), it will be changed to homogeneous point (x, y, z, 1)
    :param lidar_matrix: 4x4
    :param camera_intrinsic_matrix: camera intrinsic matrix, 3x3
    :return: camera_point_Lid2img: points in image coordinates [x, y, z]
             pixel_point: points in pixel coordinates [u, v]
    """
    K = camera_intrinsic_matrix
    lidar_point_homogeneous = np.hstack((lidar_point, 1))
    camera_point_homogeneous = np.dot(lidar_matrix, lidar_point_homogeneous)
    camera_point_Lid2img = camera_point_homogeneous[:3] / camera_point_homogeneous[3]
    pixel_point = np.dot(K, camera_point_Lid2img)
    u = pixel_point[0] / pixel_point[2]
    v = pixel_point[1] / pixel_point[2]
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
K, R_t, external_matrix = get_transform_parameters(number_session, data)
## test lidar_point to pixel coordinates.
lidar_points = np.array([23.16, 34.73, -0.625])
points_cam_coordinate, u, v = points_lid2cam(lidar_points, external_matrix, K)

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
frame = cv2.imread(os.path.join(project_path, 'measurement4/images', '000000.jpg'))
print(f'frame: {frame.shape}')
object_dict = {}
# for item in label_dict['0']:
item = label_dict['0'][2]
tmp = {}
time = item[0]
object_id = item[1]
top_x = item[2]
top_y = item[3]
width = item[4]
height = item[5]
score = item[6]
class_id = item[6]
visibility = item[7]
x_3d = item[9]
y_3d = item[10]
z_3d = item[11]
len_3d = item[12]
wid_3d = item[13]
hei_3d = item[14]
angle_3d = item[15]
tmp['time'] = item[0]
tmp['x'] = x_3d
tmp['y'] = y_3d
tmp['z'] = z_3d
tmp['l'] = len_3d
tmp['w'] = wid_3d
tmp['h'] = hei_3d
tmp['ry'] = angle_3d

# print(f'Type of tmp: {type(tmp)} \n'
#       f'tmp: {tmp}')
lidar_points = np.array([top_x, top_y, z_3d + hei_3d / 2])
number_session = 68
K, R_t, external_matrix = get_transform_parameters(number_session, data)
_, u, v = points_lid2cam(lidar_points, external_matrix, K)
print(f'u, v: {(u, v)}')
print(f'lidar_points: {lidar_points}')
cv2.circle(frame, (int(u), int(v)), 10, color, -1)
cv2.imwrite(os.path.join(save_path, f'{"%06d" % image_index}.jpg'), frame)
