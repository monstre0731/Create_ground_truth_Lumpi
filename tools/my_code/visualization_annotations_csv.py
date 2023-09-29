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
# camera/lidar info
metadata_path = os.path.join(data_path, 'measurement4/meta.json')
# result_path
save_path = os.path.join(data_path, 'measurement4/results/')
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
          '4': (128, 255, 0), '5': (255, 128, 0), '6': (0, 128, 255), '7': (128, 128, 255), '8': (128, 128, 128)}

csv_raw_file = os.path.join(data_path, 'measurement4/classifyTracks.csv')
df_raw = pd.read_csv(csv_raw_file)
total_list_raw = []
for index, row in df_raw.iterrows():
    row_list = row.tolist()
    total_list_raw.append(row_list)


## Annotations

class draw_raw:
    def __init__(self, total_list_raw):
        self.total_list_raw = total_list_raw

    def read_image(self, frame_index):
        self.frame_index = frame_index
        self.image = cv2.imread(os.path.join(data_path, f'measurement4/images_fps_30/{"%06d" % frame_index}.jpg'))

    def draw_on_image(self, row_index, rotation_matrix_lidar, tvec, K):
        annotation_index = self.total_list_raw[row_index]
        self.frame = int(total_list_raw[row_index][0] * 30)
        print(f'annotation_index: {self.frame} >>>>> \n')
        if self.frame == self.frame_index:
            self.center_point = np.array([annotation_index[9], annotation_index[10], annotation_index[11]])
            self.color = colors[str(int(annotation_index[7]))]
            self.id = annotation_index[1]
            _, u, v = points_lid2cam(self.center_point, rotation_matrix_lidar, tvec, K)
            print(f'u,v: {(u, v)}')
            draw_points(self.image, u, v, self.color, self.id)
            print(f'Add the point to the frame: {self.frame_index}')
        else:
            print(f'Not in this frame')

    def save_image(self):
        cv2.imwrite(os.path.join(save_image_path, f'{"%06d" % self.frame_index}.jpg'), self.image)
for frame_index in range(1000):
    draw_example = draw_raw(total_list_raw)
    draw_example.read_image(frame_index)
    for row_index in tqdm(range(len(total_list_raw))):
        draw_example.draw_on_image(row_index, rotation_matrix_lidar, tvec, K)
    draw_example.save_image()

csv_file = os.path.join(save_path, 'fps_30_1.csv')
df = pd.read_csv(csv_file)
total_list = []
for index, row in df.iterrows():
    row_list = row.tolist()
    total_list.append(row_list)
total_array = np.array(total_list)
frame_array = np.unique(total_array[:, 0])
frame_total_number = len(frame_array)

for frame_index in tqdm(range(frame_total_number)):
    image = cv2.imread(os.path.join(data_path, f'measurement4/images_fps_30/{"%06d" % frame_index}.jpg'))
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
