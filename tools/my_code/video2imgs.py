import cv2
import os
import sys
data_path = '/Users/qingwuliu/Documents/measurement4/cam/'
save_path = '/Users/qingwuliu/Documents/measurement4/images_fps_30'
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f'Creating save_path: {save_path}')
else:
    print(f'save_path has existed... {save_path}')

video_path = os.path.join(data_path, 'video_6.mp4')
cap = cv2.VideoCapture(video_path)

desired_fps = 30  # Replace with your desired FPS

frame_count = 0
index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % (int(cap.get(cv2.CAP_PROP_FPS)) // desired_fps) == 0:
        # Save the frame as an image
        cv2.imwrite(os.path.join(save_path, f'{"%06d.jpg" % index}'), frame)
        print(f'index: {index} | frame: {frame_count} >>>>>> \n')
        index += 1

    frame_count += 1


cap.release()
