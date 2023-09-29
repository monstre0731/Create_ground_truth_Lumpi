import cv2
import numpy as np


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


def draw_points(frame, x, y, color, object_id):
    # color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    font_thickness = 2
    text = str(object_id)

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

    camera_point_Lid2img = np.dot(rotation_matrix, lidar_point) + tvec
    # print(f'camera_point_Lid2img: {camera_point_Lid2img}')
    pixel_point = np.dot(K, camera_point_Lid2img)
    # print(f'pixel_point: {pixel_point}')

    u = pixel_point[0] / pixel_point[2]
    v = pixel_point[1] / pixel_point[2]
    # print(f'u and v: {(u, v)}')
    return camera_point_Lid2img, u, v
