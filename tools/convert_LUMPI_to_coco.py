import os

DATA_PATH = 'data/'
VIDEO_PATH = 'data/cam/video_6.mp4'
IMAGE_PATH = 'data/images/'  # video_6

import cv2
from PIL import Image

output_dir = IMAGE_PATH
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f'fps is: {fps} >>>>>>> \n')
# Initialize frame counter

frame_count = 0
index = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    # Break the loop if we have reached the end of the video
    if not ret:
        break
    # Only save every 10th frame (10 fps)
    if frame_count % 10 == 0:
        # Convert the OpenCV frame to a Pillow image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Save the image with a filename based on the frame count
        image_filename = os.path.join(IMAGE_PATH, f'{index:06d}.jpg')
        index += 1
        pil_image.save(image_filename)

    frame_count += 1

# Release the video capture object and close the output directory
cap.release()
cv2.destroyAllWindows()

print(f"Frames extracted finish, total frame: {frame_count}")
