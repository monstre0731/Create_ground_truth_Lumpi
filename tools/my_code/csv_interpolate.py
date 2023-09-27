import numpy as np
from scipy.interpolate import interp1d
import os
import sys
import pickle
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
    list_final = []
    diff_point = 0
    while mini_point <= maxi_point:
        point_added = mini_point
        list_final.append(point_added)
        mini_point += 1 / 3
    return list_final


## Read data
df = pd.read_csv(classifyTracks_path)

## Number of object_ids
object_ids = []
for item_row in tqdm(df.iloc):
    object_id = item_row["id_track"]
    if object_id not in object_ids:
        object_ids.append(object_id)

print(f'Total object: {len(object_ids)}')

# Index(['time', 'id_track', 'x_2d', 'y_2d', ' w_2d', 'h_2d', 'score',
#        'class_id', 'visibility', 'x_3d', 'y_3d', 'z_3d', 'l_3d', ' w_3d',
#        'h_3d', 'heading'],
#       dtype='object')
total_dict = {}
# for index_object in tqdm(range(len(object_ids))):
for index_object in tqdm(range(1)):
    object_id = object_ids[index_object]

    time_list = []
    x_3d_list = []
    y_3d_list = []
    z_3d_list = []
    visibility_list = []
    heading_list = []

    for item_row in tqdm(df.iloc):
        if item_row["id_track"] == object_id:
            class_id = item_row['class_id']
            l_3d = item_row['l_3d']
            w_3d = item_row['w_3d']
            h_3d = item_row['h_3d']
            time_list.append(item_row["time"])
            x_3d_list.append(item_row["x_3d"])
            y_3d_list.append(item_row["y_3d"])
            z_3d_list.append(item_row["z_3d"])
            visibility_list.append(item_row["visibility"])
            heading_list.append(item_row["heading"])

    t_ = np.array(time_list)
    x_ = np.array(x_3d_list)
    y_ = np.array(y_3d_list)
    z_ = np.array(z_3d_list)
    v_ = np.array(visibility_list)
    h_ = np.array(heading_list)
    time_interpolation = find_time_interpolation(time_list[0], time_list[-1])
    x_3d_interpolation = []
    y_3d_interpolation = []
    z_3d_interpolation = []
    v_3d_interpolation = []
    h_3d_interpolation = []

    interp_func_x = interp1d(t_, x_, kind='linear')
    interp_func_y = interp1d(t_, y_, kind='linear')
    interp_func_z = interp1d(t_, z_, kind='linear')
    interp_func_v = interp1d(t_, v_, kind='linear')
    interp_func_h = interp1d(t_, h_, kind='linear')

    threed_info_list = []
    for item_add in time_interpolation:
        if item_add <= t_[0]:
            item_add = t_[0]
        elif item_add >= t_[-1]:
            item_add = t_[-1]
        threed_list = {}
        x_inter = interp_func_x(item_add)
        y_inter = interp_func_y(item_add)
        z_inter = interp_func_z(item_add)
        v_inter = interp_func_v(item_add)
        h_inter = interp_func_h(item_add)

        x_3d_interpolation.append(x_inter)
        y_3d_interpolation.append(y_inter)
        z_3d_interpolation.append(z_inter)
        v_3d_interpolation.append(v_inter)
        h_3d_interpolation.append(h_inter)
        threed_list['frame'] = int(item_add / (1 / 3))
        threed_list['track_id'] = object_id
        threed_list['class_id'] = class_id
        threed_list['visibility'] = v_inter
        threed_list['x_3d'] = x_inter
        threed_list['y_3d'] = y_inter
        threed_list['z_3d'] = z_inter
        threed_list['l_3d'] = l_3d
        threed_list['w_3d'] = w_3d
        threed_list['h_3d'] = h_3d
        threed_list['heading'] = h_inter
        threed_info_list.append(threed_list)

    print(f'len of x_3d_final: {len(x_3d_interpolation)}')
    print(f'len of y_3d_final: {len(y_3d_interpolation)}')
    print(f'len of z_3d_final: {len(z_3d_interpolation)}')
    print(f'len of value_final: {len(v_3d_interpolation)}')
    print(f'len of value_final: {len(h_3d_interpolation)}')

    total_dict[str(index_object)] = threed_info_list

pickle_path = os.path.join(save_path, 'total_dict.pkl')
with open(pickle_path, 'wb') as pickle_file:
    pickle.dump(total_dict, pickle_file)


