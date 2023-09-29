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
data_path = '/Users/qingwuliu/Documents/'
classifyTracks_path = os.path.join(data_path, 'measurement4/classifyTracks_cp.csv')
save_path = os.path.join(data_path, 'measurement4/results/')

if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f'Creating save path, {save_path}')
else:
    print(f'save path has already existed: {save_path}')


## functions
def find_time_interpolation(value_minimum, value_maximum):
    list_tmp_mini = [i + 0.1 * int(value_minimum * 10) for i in [0, 0.033, 0.066, 0.1]]
    list_tmp_maxi = [i + 0.1 * int(value_maximum * 10) for i in [0, 0.033, 0.066, 0.1]]
    mini_point = min(list_tmp_mini, key=lambda x: abs(x - value_minimum))
    maxi_point = min(list_tmp_maxi, key=lambda x: abs(x - value_maximum))
    mini_point = min(value_minimum, mini_point)
    maxi_point = max(value_maximum, maxi_point)
    list_final = []
    diff_point = 0
    while mini_point <= (maxi_point + 0.1 / 3):
        point_added = mini_point
        list_final.append(point_added)
        mini_point += 0.1 / 3
    return list_final


csv_file = classifyTracks_path  # 替换成你的CSV文件路径
df = pd.read_csv(csv_file)

# 将每一行保存为一个列表，并将所有行存储在一个总的列表中
total_list = []
for index, row in df.iterrows():
    row_list = row.tolist()
    total_list.append(row_list)

for item in total_list:
    item.append(int(item[0] * 30))
id_list = []
for list_index in total_list:
    if list_index[1] not in id_list:
        id_list.append(list_index[1])
print(f'Num of objects: {len(id_list)}')

keys = ['time', 'id_track', 'x_2d', 'y_2d', ' w_2d', 'h_2d', 'score', 'class_id', 'visibility', 'x_3d', 'y_3d', 'z_3d',
        'l_3d', 'w_3d', 'h_3d', 'heading']
class row_labels:
    def __init__(self, total_list, index):
        self.info = total_list[index]
        self.time = total_list[index][0]
        self.id_track = total_list[index][1]
        self.x_2d = total_list[index][2]
        self.y_2d = total_list[index][3]
        self.w_2d = total_list[index][4]
        self.h_2d = total_list[index][5]
        self.score = total_list[index][6]
        self.class_id = total_list[index][7]
        self.visibility = total_list[index][8]
        self.x_3d = total_list[index][9]
        self.y_3d = total_list[index][10]
        self.z_3d = total_list[index][11]
        self.l_3d = total_list[index][12]
        self.w_3d = total_list[index][13]
        self.h_3d = total_list[index][14]
        self.heading = total_list[index][15]
        self.fps = 30
        self.frame = int(self.time * self.fps)
        self.total = [self.frame] + self.info


csv_filename = os.path.join(save_path, "fps_30_1.csv")
total_list_inter = []
with open(csv_filename, 'a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for index_id in tqdm(id_list):
        object_list = []
        for index_row in range(len(total_list)):
            row_current = row_labels(total_list, index_row)
            if row_current.id_track == index_id:
                object_list.append(row_current.total)

        ## interpolate
        object_numpy = np.array(object_list)
        x = object_numpy[:, 0]
        y = object_numpy[:, 1:]
        new_x = np.arange(max(x) + 1)

        interpolated_values = []
        for col in y.T:
            interpolated_col = np.interp(new_x, x, col)
            # print(f'shape: {interpolated_col.shape}')
            interpolated_values.append(interpolated_col)
        # Stack the interpolated columns horizontally to form the final array
        interpolated_array = np.column_stack(interpolated_values)
        new_x_reshaped = new_x.reshape(-1, 1)
        data_current_id = np.hstack((new_x_reshaped, interpolated_array))

        data_current_list = data_current_id.tolist()
        for row in data_current_list:
            total_list_inter.append(row)
        # print(f'Length of data_current_id: {len(data_current_id)} >>>>> \n')
        # arr = data_current_id
        # max_decimal_places = max([len(str(round(elem, 3)).split('.')[1]) if isinstance(elem, float) else 0 for elem in arr.flatten()])
        # # 使用 np.around() 将所有元素四舍五入到最大小数位数
        # formatted_arr = np.around(arr, max_decimal_places)
        # # data_current_id_2_after_point = np.array([float(f"{x:.3f}") for x in data_current_id])
        # csv_writer.writerows(formatted_arr)

print(f'finish writing')






