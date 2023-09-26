import cv2
import numpy as np


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
