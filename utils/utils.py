import cv2
import numpy as np
import math

def draw_box(img,img_coor, show_num):
    #color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    colors = [(255,0,0),(0,255,0),(0,0,255),(0,0,0), (255,222,0), (255,120,0), (255,0,120), (0,120,120)]
    color = (255, 255, 255)
    vs = []

    for i in range(8):
        vs.append((int(img_coor[i,0]),int(img_coor[i,1])))
    if show_num:
        for i in range(8):
            cv2.putText(img,'%d'%i,vs[i] ,cv2.FONT_HERSHEY_PLAIN, 2,colors[i],2)
    

    cv2.line(img,vs[0],vs[1],color,2,8,0)
    cv2.line(img,vs[1],vs[2],color,2,8,0)
    cv2.line(img,vs[2],vs[3],color,2,8,0)
    cv2.line(img,vs[3],vs[0],color,2,8,0)

    cv2.line(img,vs[4],vs[5],color,2,8,0)
    cv2.line(img,vs[5],vs[6],color,2,8,0)
    cv2.line(img,vs[6],vs[7],color,2,8,0)
    cv2.line(img,vs[7],vs[4],color,2,8,0)

    cv2.line(img,vs[0],vs[4],color,2,8,0)
    cv2.line(img,vs[3],vs[7],color,2,8,0)
    cv2.line(img,vs[2],vs[6],color,2,8,0)
    cv2.line(img,vs[1],vs[5],color,2,8,0)


def draw_box_1(img,img_coor, show_num):
    #color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    colors = [(255,0,0),(0,255,0),(0,0,255),(0,0,0), (255,222,0), (255,120,0), (255,0,120), (0,120,120)]
    color = (0, 255, 0)
    vs = []

    for i in range(8):
        vs.append((int(img_coor[i,0]),int(img_coor[i,1])))
    if show_num:
        for i in range(8):
            cv2.putText(img,'%d'%i,vs[i] ,cv2.FONT_HERSHEY_PLAIN, 2,colors[i],2)

    cv2.line(img,vs[0],vs[1],color,2,8,0)
    cv2.line(img,vs[1],vs[2],color,2,8,0)
    cv2.line(img,vs[2],vs[3],color,2,8,0)
    cv2.line(img,vs[3],vs[0],color,2,8,0)

    cv2.line(img,vs[4],vs[5],color,2,8,0)
    cv2.line(img,vs[5],vs[6],color,2,8,0)
    cv2.line(img,vs[6],vs[7],color,2,8,0)
    cv2.line(img,vs[7],vs[4],color,2,8,0)

    cv2.line(img,vs[0],vs[4],color,2,8,0)
    cv2.line(img,vs[3],vs[7],color,2,8,0)
    cv2.line(img,vs[2],vs[6],color,2,8,0)
    cv2.line(img,vs[1],vs[5],color,2,8,0)

def anno_dict(annotation_list):
    anno_list = {'frame_id':annotation_list[0], 
                'time': annotation_list[1],
                'obj_id': annotation_list[2],
                'x_2d': annotation_list[3],
                'y_2d' :annotation_list[4],
                'w_2d' : annotation_list[5],
                'h_2d' : annotation_list[6],
                'score' : annotation_list[7],
                'class_id' : annotation_list[8],
                'visibility' : annotation_list[9],
                'x_3d' :annotation_list[10],
                'y_3d' : annotation_list[11],
                'z_3d' : annotation_list[12],
                'l_3d' : annotation_list[13],
                'w_3d' : annotation_list[14],
                'h_3d' : annotation_list[15],
                'heading_3d' : annotation_list[16]}
    return anno_list



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

def compute_box_3d(anno_list):

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
    R = rotz(anno_list['heading_3d'])
    # R = rotz(0)
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
    

def draw_lidar_points(frame, lidar_point, color_id, object_id, camera_info):
    '''
    This is for draw several lidar points on image plane
    frame: readed image
    lidar_point: np.array([x, y, z])
    color_id: 1, 2, 3, ..., 8
    object_id: int
    camera_info: rotation_matrix_lidar, tvec, K, frame
    '''
    rotation_matrix_lidar = camera_info['rotation_matrix_lidar']
    K = camera_info['K']
    R_t = camera_info['R_t']
    tvec = camera_info['tvec']

    colors = {'1': (0, 255, 0), '2': (255, 0, 0), '3': (0, 0, 255),
          '4': (128, 255, 0), '5': (255, 128, 0), '6': (0, 128, 255), '7': (128, 128, 255), '8': (0, 128, 128)}
    _, x, y = points_lid2cam(lidar_point=lidar_point, rotation_matrix=rotation_matrix_lidar, tvec=tvec, camera_intrinsic_matrix=K)
    color = colors[str(color_id)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    font_thickness = 2
    text = str(object_id)
    # 在图像上添加文本
    cv2.putText(frame, text, (int(x-1), int(y - 1)), font, font_scale, font_color, font_thickness)
    cv2.circle(frame, (int(x), int(y)), 10, color, -1)

def draw_points(frame, x, y, color, object_id):
    # color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # 白色
    font_thickness = 2
    text = object_id
    print(f'text: {text}')

    # 在图像上添加文本
    cv2.putText(frame, text, (int(x-1), int(y - 1)), font, font_scale, font_color, font_thickness)
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

    camera_point = np.dot(rotation_matrix, lidar_point) + tvec
    # print(f'camera_point_Lid2img: {camera_point_Lid2img}')
    pixel_point = np.dot(K, camera_point)
    # print(f'pixel_point: {pixel_point}')

    u = pixel_point[0] / pixel_point[2]
    v = pixel_point[1] / pixel_point[2]
    # print(f'u and v: {(u, v)}')
    return camera_point, u, v

def points_cam2pixel(camera_point_lid2img, camera_intrinsic_matrix):
    K = camera_intrinsic_matrix
    pixel_point = np.dot(K, camera_point_lid2img)
    # print(f'pixel_point: {pixel_point}')

    u = pixel_point[0] / pixel_point[2]
    v = pixel_point[1] / pixel_point[2]
    # print(f'u and v: {(u, v)}')
    return camera_point_lid2img, u, v

def calculate_distance(point1, point2):
    x1, y1, z1 = point1[0], point1[1], point1[2]
    x2, y2, z2 = point2[0], point2[1], point2[2]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

# w_3d_cam = calculate_distance(camera_points_list[0], camera_points_list[1])
# l_3d_cam = calculate_distance(camera_points_list[1], camera_points_list[2])
# h_3d_cam = calculate_distance(camera_points_list[0], camera_points_list[4])