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
annotation_path = os.path.join(measurement_path, 'results/total_inter.csv')
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

# draw 3d bouding boxes
for frame_index in tqdm(range(20)):
    # frame_index = 1
    image = cv2.imread(os.path.join(project_path, f'LUMPI_new/data_example/{"%06d" % frame_index}.jpg'))
    for item in total_list:
        # item = total_list[6]
        if int(item[0]) == frame_index:
        # if int(item[0]) == frame_index and int(item[2]) <= 6:
            anno_list0 = anno_dict(item)
            lidar_points = compute_box_3d(anno_list0)
            lidar_point_center = np.array([anno_list0['x_3d'],anno_list0['y_3d'], anno_list0['z_3d'] ])
            camera_point_center, _, _ = points_lid2cam(lidar_point=lidar_point_center, rotation_matrix=rotation_matrix_lidar, tvec=tvec, camera_intrinsic_matrix=K)
            
            # cv2.circle(image, (int(u_center), int(v_center)), 10, [0, 255, 255], -1)
            # lidar_center
            draw_lidar_points(frame=image, lidar_point=lidar_point_center, color_id=1, object_id=8, camera_info=camera_info)
            
            uv_list = []

            camera_point_list = []
            for lidar_point in lidar_points:
                camera_point, x, y = points_lid2cam(lidar_point=lidar_point, rotation_matrix=rotation_matrix_lidar, tvec=tvec, camera_intrinsic_matrix=K)
                tmp = np.array([int(x), int(y)])
                uv_list.append(tmp)
                if int(y) < 0:
                    uv_list = []
                    continue
                
                camera_point_list.append(camera_point)
            if len(uv_list) == 8:
                
                uv_array = np.array(uv_list)
                # # draw 3d bounding box
                draw_box(image, uv_array, False)
                min_u = min(uv_array[:, 0])
                max_u = max(uv_array[:, 0])
                min_v = min(uv_array[:, 1])
                max_v = max(uv_array[:,1])
                w_2d = max_u - min_u
                h_2d = max_v - min_v
                color = (255, 255, 0) 
                thickness = 2  
            # cv2.rectangle(image, (int(min_u), int(min_v)), (int(max_u), int(max_v)), color, thickness)
    cv2.imwrite(os.path.join(result_path, f'{frame_index}.jpg'), image)


# draw 3d bouding boxes with camera coordinate
for frame_index in tqdm(range(20)):
    # frame_index = 1
    # image = cv2.imread(os.path.join(project_path, f'LUMPI_new/data_example/{"%06d" % frame_index}.jpg'))
    image = cv2.imread(os.path.join(result_path, f'{frame_index}.jpg'))
    for item in total_list:
        # if int(item[0]) == frame_index:
        if int(item[0]) == frame_index and int(item[2]) <= 6:
            anno_list0 = anno_dict(item)
            lidar_points = compute_box_3d(anno_list0)
            camera_points_list_lidar = []
            for lidar_point in lidar_points:
                camera_point, _, _ = points_lid2cam(lidar_point=lidar_point, rotation_matrix=rotation_matrix_lidar, tvec=tvec, camera_intrinsic_matrix=K)
                camera_points_list_lidar.append(camera_point)
            # calculate angle

            # give the center x, y, z to a new one
            anno_list_cam = anno_list0.copy()

            lidar_point_center = np.array([anno_list0['x_3d'],anno_list0['y_3d'], anno_list0['z_3d'] ])
            camera_points_center, u_center, v_center = points_lid2cam(lidar_point=lidar_point_center, rotation_matrix=rotation_matrix_lidar, tvec=tvec, camera_intrinsic_matrix=K)
            # cv2.circle(image, (int(u_center), int(v_center)), 10, [0, 255, 255], -1)
            # lidar_center
            draw_lidar_points(frame=image, lidar_point=lidar_point_center, color_id=1, object_id=8, camera_info=camera_info)
            
            # Replace lidar 3d information with camera information
            anno_list_cam['x_3d'] = camera_points_center[0]
            anno_list_cam['y_3d'] = camera_points_center[1]
            anno_list_cam['z_3d'] = camera_points_center[2]
            anno_list_cam['w_3d'] = anno_list0['h_3d']
            anno_list_cam['h_3d'] = anno_list0['w_3d']
            # anno_list_cam['heading_3d'] = heading_3d_came

            camera_tmp = compute_box_3d_cam(anno_list=anno_list_cam, rot_angle = 1.0)
        
            uv_list_real = []
            camera_point_list_real = []
            for camera_point_tmp in camera_tmp:
                camera_points, x, y = points_cam2pixel(camera_point_tmp, K)
                tmp = np.array([int(x), int(y)])
                uv_list_real.append(tmp)
                if int(y) < 0:
                    uv_list = []
                    continue
                camera_point_list_real.append(camera_points)

            if len(uv_list_real) == 8:
                uv_array = np.array(uv_list_real)
                # # draw 3d bounding box
                draw_box_1(image, uv_array, False)
                # draw 2d bounding box
            # cv2.rectangle(image, (int(min_u), int(min_v)), (int(max_u), int(max_v)), color, thickness)
    cv2.imwrite(os.path.join(result_path, f'{frame_index}.jpg'), image)



# save 2d bounding boxes into annotation.csv
# total_list_real = []
# for anno_list_item in tqdm(total_list):
#     anno_list0 = anno_dict(anno_list_item)
#     lidar_points = compute_box_3d(anno_list0)
#     uv_list = []
#     for lidar_point in lidar_points:
#         _, x, y = points_lid2cam(lidar_point=lidar_point, rotation_matrix=rotation_matrix_lidar, tvec=tvec, camera_intrinsic_matrix=K)
#         tmp = np.array([int(x), int(y)])
#         uv_list.append(tmp)
#     uv_array = np.array(uv_list)
#     # draw_box(image, uv_array)
#     min_u = min(uv_array[:, 0])
#     max_u = max(uv_array[:, 0])
#     min_v = min(uv_array[:, 1])
#     max_v = max(uv_array[:,1])
#     w_2d = max_u - min_u
#     h_2d = max_v - min_v
#     anno_list0['x_2d'] = min_u
#     anno_list0['y_2d'] = min_v
#     anno_list0['w_2d'] = w_2d
#     anno_list0['h_2d'] = h_2d
#     tmp = []
#     for key in anno_list0.keys():
#         tmp.append(anno_list0[key])
#     total_list_real.append(tmp)
# annotation_data_array = np.array(total_list_real)
# save_path = '/Users/qingwuliu/Documents/Code/LUMPI_new/LUMPI_new/results/annotation'
# csv_file_path = os.path.join(save_path, f"annotation.csv")
# with open(csv_file_path, mode='w', newline='') as file:
#     csv_writer = csv.writer(file)
#     csv_writer.writerows(annotation_data_array)


def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def compute_box_3d_cam(anno_list, rot_angle):

    #### LiDAR coordinate, rotate with y
    '''.DS_Store
        ^ z   
        |
        |
        |
        . - - - - - - - > x
       /
      /
     /
    v y
    
    '''
    # todo
    R = roty(anno_list['heading_3d'])
    R = roty(rot_angle)
    l = anno_list['l_3d']
    w = anno_list['w_3d']
    h = anno_list['h_3d']

    x_3d = anno_list['x_3d']
    y_3d = anno_list['y_3d']
    z_3d = anno_list['z_3d']

    # 3d bounding box corners
    #            0     1      2    3      4     5    6     7
    x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    z_corners = [h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # add the center
    corners_3d[0, :] = corners_3d[0, :] + x_3d
    corners_3d[1, :] = corners_3d[1, :] + y_3d
    corners_3d[2, :] = corners_3d[2, :] + z_3d

    return np.transpose(corners_3d)

def compute_box_3d_cam(anno_list, rot_angle):

    #### LiDAR coordinate, rotate with y
    '''.DS_Store
        ^ z   
        |
        |
        |
        . - - - - - - - > x
       /
      /
     /
    v y
    
    '''
    # todo
    R = roty(rot_angle)
    l = anno_list['l_3d']
    w = anno_list['w_3d']
    h = anno_list['h_3d']

    x_3d = anno_list['x_3d']
    y_3d = anno_list['y_3d']
    z_3d = anno_list['z_3d']

    # 3d bounding box corners
    #            0     1      2    3      4     5    6     7
    x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    z_corners = [h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # add the center
    corners_3d[0, :] = corners_3d[0, :] + x_3d
    corners_3d[1, :] = corners_3d[1, :] + y_3d
    corners_3d[2, :] = corners_3d[2, :] + z_3d

    return np.transpose(corners_3d)






def compute_box_3d_cam_real(anno_list):

    #### LiDAR coordinate, rotate with y
    '''.DS_Store
        ^ z   
        |
        |
        |
        . - - - - - - - > x
       /
      /
     /
    v y
    
    '''
    # todo
    R = roty(anno_list['heading_3d'])
    l = anno_list['l_3d']
    w = anno_list['w_3d']
    h = anno_list['h_3d']

    x_3d = anno_list['x_3d']
    y_3d = anno_list['y_3d']
    z_3d = anno_list['z_3d']

    # 3d bounding box corners
    #            0     1      2    3      4     5    6     7
    x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    z_corners = [h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # add the center
    corners_3d[0, :] = corners_3d[0, :] + x_3d
    corners_3d[1, :] = corners_3d[1, :] + y_3d
    corners_3d[2, :] = corners_3d[2, :] + z_3d

    return np.transpose(corners_3d)




def calculate_headings(camera_points_list):

    x1, y1 = camera_points_list[0][0],  camera_points_list[0][2]
    x2, y2 = camera_points_list[3][0],  camera_points_list[3][2] 
    x3, y3 = camera_points_list[4][0],  camera_points_list[4][2] 
    x4, y4 = camera_points_list[7][0],  camera_points_list[7][2]
    # 计算矩形中心点坐标
    x_center = (x1 + x2 + x3 + x4) / 4
    y_center = (y1 + y2 + y3 + y4) / 4
    # 计算长轴和短轴的长度
    long_axis = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    short_axis = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    # 计算旋转角度（以弧度为单位）
    rotation_angle = math.atan2((y2 - y1), (x2 - x1))
    return rotation_angle

heading_angle = calculate_headings(camera_points_list=camera_points_list)


def calculate_headings_new(camera_points_list):

    x1, y1 = camera_points_list[0][0],  camera_points_list[0][2]
    x2, y2 = camera_points_list[1][0],  camera_points_list[1][2] 
    x3, y3 = camera_points_list[2][0],  camera_points_list[2][2] 
    x4, y4 = camera_points_list[3][0],  camera_points_list[3][2]
    # 计算矩形中心点坐标
    x_center = (x1 + x2 + x3 + x4) / 4
    y_center = (y1 + y2 + y3 + y4) / 4
    # 计算长轴和短轴的长度
    long_axis = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    short_axis = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    # 计算旋转角度（以弧度为单位）
    rotation_angle = math.atan2((y2 - y1), (x2 - x1))
    return rotation_angle

heading_angle = calculate_headings_new(camera_points_list=camera_points_list)


def calculate_heading(camera_points_center, camera_points_list_lidar, camera_points_list):
    # 坐标原点
    x_0, y_0 = camera_points_center[0], camera_points_center[2]
    # 相机坐标下
    x_1, y_1 = camera_points_list[1][0], camera_points_list[1][2]
    # 目标，lidar坐标下
    x_2, y_2 = camera_points_list_lidar[0][0], camera_points_list_lidar[0][2]
    # print(f'坐标原点：{(x_0, y_0)}')
    # print(f'相机坐标：{(x_1, y_1)}')
    # print(f'目标坐标：{(x_2, y_2)}')

    # 计算相对于原点的坐标
    x1_relative, y1_relative = x_1 - x_0, y_1 - y_0
    x2_relative, y2_relative = x_2 - x_0, y_2 - y_0

    # 计算旋转角度（弧度制）
    rotation_angle = math.atan2(y2_relative - y1_relative, x2_relative - x1_relative)

    # 输出旋转角度
    print("相机：旋转角度（弧度制）为:", rotation_angle)
    return rotation_angle



rotation_angle = calculate_heading(camera_points_center, camera_points_list_lidar, camera_points_list)



def calculate_rotation_angle(camera_points_center, camera_points_list_lidar, camera_points_list):
    # 坐标原点
    x_0, y_0 = camera_points_center[0], camera_points_center[2]
    # 相机坐标下
    x_1, y_1 = camera_points_list[1][0], camera_points_list[1][2]
    # 目标，lidar坐标下
    x_2, y_2 = camera_points_list_lidar[0][0], camera_points_list_lidar[0][2]
    angle1 = math.atan2(y2 - y0, x2 - x0)
    angle2 = math.atan2(y1 - y0, x1 - x0)
    rotation_angle = angle1 - angle2
    return rotation_angle

# # 示例使用
# x0, y0 = camera_points_center[0], camera_points_center[2] # 坐标原点
# x1, y1 = camera_points_list[1][0], camera_points_list[1][2]  # 初始坐标
# x2, y2 = camera_points_list_lidar[0][0], camera_points_list_lidar[0][2]  # 最终坐标

rotation_angle = calculate_rotation_angle(camera_points_center, camera_points_list_lidar, camera_points_list)

print("旋转弧度: ", rotation_angle)