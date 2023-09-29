import os
import sys
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm

## project_path
project_path = '/'
sys.path.append(project_path)

## data path
data_path = '/scratch2/liuqin/dataset/LUMPI'
classifyTracks_path = os.path.join(data_path, 'measurement4/classifyTracks_cp.csv')
save_path = os.path.join(data_path, 'measurement4/results/')

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f'Creating save path, {save_path}')
else:
    print(f'save path has already existed: {save_path}')


keys = ['time', 'id_track', 'x_2d', 'y_2d', 'w_2d', 'h_2d', 'score', 'class_id', 'visibility', 'x_3d', 'y_3d', 'z_3d',
        'l_3d', 'w_3d', 'h_3d', 'heading']

csv_file = classifyTracks_path
df = pd.read_csv(csv_file)


total_list = []
for index, row in df.iterrows():
    row_list = row.tolist()
    row_list.insert(0, row_list[0] * 30)
    total_list.append(row_list)

total_array = np.array(total_list)
total_object_id = np.unique(total_array[:, 2])
csv_file_path = os.path.join(save_path, f"total_inter_1.csv")


total_results = []
for object_id_index in total_object_id:

    list_obj_id = []
    print(f'current object id is: {object_id_index}')
    for row_item in total_list:
        if row_item[2] == object_id_index:
            list_obj_id.append(row_item)
    print(f'length of list_obj_id: {len(list_obj_id)}')

    array_obj_id = np.array(list_obj_id)
    frame_original = array_obj_id[:, 0]
    min_frame, max_frame = min(frame_original), max(frame_original)
    frame_new = np.arange(int(min_frame), int(max_frame) + 1)
    values_original = array_obj_id[:, 1:]

    interpolated_values = []
    for col in values_original.T:
        interpolated_col = np.interp(frame_new, frame_original, col)
        interpolated_values.append(interpolated_col)

    # Stack the interpolated columns horizontally to form the final array
    interpolated_array = np.column_stack(interpolated_values)
    new_x_reshaped = frame_new.reshape(-1, 1)
    data_current_id = np.hstack((new_x_reshaped, interpolated_array))

    list_info_after_inter = data_current_id.tolist()

    for row in data_current_id:
        total_results.append(row)
annotation_data_array = np.array(total_results)
annotation_data_by_frame = annotation_data_array[annotation_data_array[:, 0].argsort()]
save_path = os.path.join(data_path, 'measurement4/results/')
csv_file_path = os.path.join(save_path, f"total_inter.csv")

with open(csv_file_path, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(annotation_data_by_frame)


