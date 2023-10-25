import sys
project_path = '/Users/qingwuliu/Documents/Code/LUMPI_new/'
sys.path.append(project_path)
import os 
import numpy as np
import pandas as pd
import json
import cv2
import csv
from tqdm import tqdm
from utils.utils import draw_points, draw_lidar_points, get_transform_parameters, points_lid2cam, compute_box_3d, anno_dict,draw_box, draw_box_1, points_cam2pixel

## calib directly read the meta: rvec, tvec, extrinstic, intrinstic etc
## Note Lidar coordinate is the same as world coordinate 

## Take measurement 4 as example, 
#  two videos with fps as 30 Hz are captured, 14 mins
# depth should be in image coordinate 

# Problem: how to get the 2D bounding boxes?  

# data_path
measurement_path = '/Users/qingwuliu/Documents/measurement4'
annotation_path = os.path.join(measurement_path, 'results/total_inter_10_11.csv')
metadata_path = os.path.join(measurement_path, 'meta.json')
result_path = os.path.join(project_path, 'LUMPI_new/results/output')
if not os.path.exists(result_path):
    os.makedirs(result_path)

# basic info
number_session = 68

with open(metadata_path, 'r') as json_file:
    data = json.load(json_file)

K, R_t, rotation_matrix_lidar, tvec = get_transform_parameters(number_session, data)

camera_info = {'K': K, 'R_t': R_t, 'rotation_matrix_lidar': rotation_matrix_lidar, 'tvec': tvec}

## Read inter_total.csv
df = pd.read_csv(annotation_path)

total_list = []
for index, row in df.iterrows():
    row_list = row.tolist()
    total_list.append(row_list[:-1])

total_array = np.array(total_list)
frame_array = np.unique(total_array[:, 0])
frame_total_number = len(frame_array)
# convert each row into the anno_dict
index = 0
current_row_as_dict = anno_dict(total_list[index])
frame_index_list = [i for i in range(24691)]


# path 
save_path = os.path.join(project_path, 'LUMPI_new/results')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 初始化以frame为index的字典
total_frame_dict= {}
for frame_index in frame_index_list:
    total_frame_dict[f'{frame_index}'] = []

# Modify the bb2d and bb3d
for item in tqdm(total_list):
    # save txt file direction
    # item = total_list[0]
    # represent as anno_dict
    current_row_as_dict = anno_dict(item)
    # change bb2d, bb3d infos
    new_row_as_dict = current_row_as_dict.copy()

    lidar_points = compute_box_3d(current_row_as_dict)
    lidar_point_center = np.array([current_row_as_dict['x_3d'],current_row_as_dict['y_3d'], current_row_as_dict['z_3d'] ])
    camera_point_center, _, _ = points_lid2cam(lidar_point=lidar_point_center, rotation_matrix=rotation_matrix_lidar, tvec=tvec, camera_intrinsic_matrix=K)   
    
    uv_list = []
    for lidar_point in lidar_points:
        camera_point, x, y = points_lid2cam(lidar_point=lidar_point, rotation_matrix=rotation_matrix_lidar, tvec=tvec, camera_intrinsic_matrix=K)
        tmp = np.array([int(x), int(y)])
        uv_list.append(tmp)
        if int(y) < 10:
            uv_list = []
            continue
        if len(uv_list) == 8:
            
            uv_array = np.array(uv_list)
            # # draw 3d bounding box
            # draw_box(image, uv_array, False)
            min_u = min(uv_array[:, 0])
            max_u = max(uv_array[:, 0])
            min_v = min(uv_array[:, 1])
            max_v = max(uv_array[:,1])
            if min_u < 0 or min_v < 0 or max_u > 1632 or max_v > 1216:
                # print(f'min_u is smaller than 0: {min_u} >>>\n')
                continue
            else:
                new_row_as_dict['l_3d'] = current_row_as_dict['l_3d']
                new_row_as_dict['h_3d'] = current_row_as_dict['w_3d']
                new_row_as_dict['w_3d'] = current_row_as_dict['h_3d']
                new_row_as_dict['left_2d'] = min_u
                new_row_as_dict['top_2d'] = min_v
                new_row_as_dict['right_2d'] = max_u
                new_row_as_dict['bottom_2d'] = max_v
                new_row_as_dict['x_3d_cam'], new_row_as_dict['y_3d_cam'], new_row_as_dict['z_3d_cam'] = camera_point_center[0], camera_point_center[1], camera_point_center[2]
                
                new_list = [int(new_row_as_dict['frame_id']), int(new_row_as_dict['obj_id']), int(new_row_as_dict['class_id']),
                            new_row_as_dict['visibility'], new_row_as_dict['left_2d'], new_row_as_dict['top_2d'], 
                            new_row_as_dict['right_2d'],  new_row_as_dict['bottom_2d'], 
                            new_row_as_dict['x_3d'], new_row_as_dict['y_3d'], new_row_as_dict['z_3d'],
                            new_row_as_dict['l_3d'], new_row_as_dict['h_3d'], new_row_as_dict['w_3d'], new_row_as_dict['heading_3d'], 
                            new_row_as_dict['x_3d_cam'], new_row_as_dict['y_3d_cam'], new_row_as_dict['z_3d_cam']]               
                # print(f'new_list: {new_list}')
                total_frame_dict[str(int(current_row_as_dict['frame_id']))].append(new_list)

       
for item in tqdm(frame_index_list):
    # print(f'item: {item}')
    with open(os.path.join(save_path, f'label/{"%06d.txt" % int(item)}'), 'w') as file:
        for num_index in range(len(total_frame_dict[str(item)])):
            values_as_string = ' '.join(map(str, total_frame_dict[str(item)][num_index]))
            file.write(values_as_string)
            file.write("\n")
print(f'Please find the txt files in: {os.path.join(save_path, "label")}')


def test_visualization_2d_LUMPI(project_path):
    ## test on some images
    for frame_index in range(20):
        # frame_index = 4
        result_path_1 = os.path.join(project_path, 'LUMPI_new/results/output_1')
        if not os.path.exists(result_path_1):
            os.makedirs(result_path_1)
        image = cv2.imread(os.path.join(project_path, f'LUMPI_new/data_example/{"%06d" % frame_index}.jpg'))
        for label in total_frame_dict[str(frame_index)]:
            min_u, min_v, max_u, max_v = label[4], label[5], label[6], label[7]
            # print(f'left, top, right, bottom: {(min_u, min_v, max_u, max_v)}')
            color = (255, 255, 0) 
            thickness = 2  
            cv2.rectangle(image, (int(min_u), int(min_v)), (int(max_u), int(max_v)), color, thickness)
        cv2.imwrite(os.path.join(result_path_1, f'{frame_index}.jpg'), image)

