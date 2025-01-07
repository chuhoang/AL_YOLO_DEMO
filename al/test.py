# from main import Active_Learning
# import glob
# al = Active_Learning('/home/mq/data_disk2T/Thang/bak/src/runs/detect/train10/weights/best.pt')
# al.__load_model__(3)
# a = glob.glob('/home/mq/data_disk2T/Thang/bak/src/data1/val/images/*.jpg')
# al('/home/mq/data_disk2T/Thang/bak/src/data1/data.yaml', a)
# import os
# import glob
# import shutil
# a = glob.glob('/home/mq/data_disk2T/HoangChu/JSON2YOLO/gt_dir/images/val/*.jpg')

# count = 0
# for i in a:
#     if count > 100: break
#     if os.path.exists(i.replace('.jpg', '.txt').replace('images', 'labels')):
#         count += 1
#         shutil.copy(i, '/home/mq/data_disk2T/HoangChu/JSON2YOLO/gt_dir/val1/images')
#         shutil.copy(i.replace('.jpg', '.txt').replace('images', 'labels'), '/home/mq/data_disk2T/HoangChu/JSON2YOLO/gt_dir/val1/labels')

import cv2
import numpy as np

# Define the image dimensions (for example, 640x480)
# image_width, image_height = 640, 480

# Create a blank image (you can also load an actual image using cv2.imread)
# image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
# read txt ground truth and return an array of bboxes
def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        line = line.strip().split(' ')
        class_id, x, y, w, h = [float(x) for x in line]

        bboxes.append([x, y, w, h])
    return np.array(bboxes)

def visualize(bboxes, image, class_id, out_path, color=(0, 255, 0)):
# Loop over each bounding box

    for x, y, w, h in bboxes:
        # Convert normalized YOLO coordinates to pixel values
        x_min = int((x - w / 2) )
        y_min = int((y - h / 2) )
        x_max = int((x + w / 2) )
        y_max = int((y + h / 2) )
        # Choose a color for each class

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Add the class label
        label = f"Class {class_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y_min = max(y_min, label_size[1] + 10)
        cv2.rectangle(image, (x_min, y_min - label_size[1] - 10), (x_min + label_size[0], y_min), color, -1)
        cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the image with bounding boxes
    cv2.imwrite(out_path, image)
    # cv2.imshow("YOLO Bounding Boxes", image)
    # cv2.waitKey(0)
# cv2.destroyAllWindows()

