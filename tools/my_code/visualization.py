import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

## project path and add it into sys path ## Data path
project_path = '/Users/qingwuliu/Documents/Code/LUMPI_new/'
data_path = '/Users/qingwuliu/Documents/measurement4/'
sys.path.append(project_path)
from utils.utils import draw_points, get_transform_parameters, points_lid2cam



# camera/lidar info
metadata_path = os.path.join(data_path, 'meta.json')
# result_path
save_path = os.path.join(data_path, 'results/')
# save_image_path
save_image_path = os.path.join(save_path, 'images/')
# Create path
path_need_to_create = [save_path, save_image_path]
for path_item in path_need_to_create:
    if not os.path.exists(path_item):
        os.makedirs(path_item)
        print(f'Creating save path, {path_item}')
    else:
        print(f'save path has already existed: {path_item}')

#############################################
# *****************************************
#############################################
# Basic infomation
## frame rate
fps_lidar = 10
fps_camera = 30
number_session = 68
## Camera parameters
with open(metadata_path, 'r') as json_file:
    data = json.load(json_file)
K, R_t, rotation_matrix_lidar, tvec = get_transform_parameters(number_session, data)
colors = {'1': (0, 255, 0), '2': (255, 0, 0), '3': (0, 0, 255),
          '4': (128, 255, 0), '5': (255, 128, 0), '6': (0, 128, 255), '7': (128, 128, 255), '8': (0, 128, 128)}

## Annotations
csv_file = os.path.join(save_path, 'total_inter.csv')
df = pd.read_csv(csv_file)
total_list = []
for index, row in df.iterrows():
    row_list = row.tolist()
    total_list.append(row_list)
total_array = np.array(total_list)
frame_array = np.unique(total_array[:, 0])
frame_total_number = len(frame_array)

for frame_index in tqdm(range(20)):
    print(f'frame_index: {frame_index}')
    image = cv2.imread(os.path.join(data_path, f'images_fps_30/{"%06d" % frame_index}.jpg'))
    for label_index in range(len(total_list)):
        annotation_index = total_list[label_index]
        if int(annotation_index[0]) == frame_index:
            object_id_index = annotation_index[2]
            color = colors[str(int(annotation_index[8]))]
            center_point = np.array([annotation_index[10], annotation_index[11], annotation_index[12]])
            _, u, v = points_lid2cam(center_point, rotation_matrix_lidar, tvec, K)
            draw_points(image, u, v, color, object_id_index)

    cv2.imwrite(os.path.join(save_image_path, f'{"%06d" % frame_index}.jpg'), image)
    print(f'Please find {"%06d.jpg" % frame_index} in {save_path}')

## visualization the points in camera coordinates
for frame_index in tqdm(range(21)):
    print(f'frame_index: {frame_index}')
    image = cv2.imread(os.path.join('/Users/qingwuliu/Documents/Code/LUMPI_new/LUMPI_new/data_example', f'{"%06d" % frame_index}.jpg'))
    for label_index in range(len(total_list)):
        annotation_index = total_list[label_index]
        if int(annotation_index[0]) == frame_index:
            object_id_index = annotation_index[2]
            object_x = annotation_index[3]
            object_y = annotation_index[4]
            object_z = annotation_index[5]
            color = colors[str(int(annotation_index[8]))]
            center_point = np.array([annotation_index[10], annotation_index[11], annotation_index[12]])
            cam_point, u, v = points_lid2cam(center_point, rotation_matrix_lidar, tvec, K)
            # object_text = f'({np.round(object_x, 1)},{np.round(object_y, 1)})'
            object_text = f'({np.round(cam_point[0], 1)},{np.round(cam_point[1], 1)}, {np.round(cam_point[2], 1)})'
            draw_points(image, u, v, color, object_text)
    save_image_path = '/Users/qingwuliu/Documents/Code/LUMPI_new/LUMPI_new/results'
    cv2.imwrite(os.path.join(save_image_path, f'{"%06d" % frame_index}.jpg'), image)
    print(f'Please find {"%06d.jpg" % frame_index} in {save_image_path}')