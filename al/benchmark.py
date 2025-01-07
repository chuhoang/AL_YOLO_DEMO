import os
import cv2
import glob
import torch 
import numpy as np
from ultralytics import YOLO
from joblib import dump, load
from ood.pyod.pyod.models.lunar import LUNAR
from feat_extract import infer_cropped_image, extract_feature, init_model

def read_txt_bbox(img_path, class_id):

    img = cv2.imread(img_path)
    with open(img_path.replace('images/', 'labels/').replace('.jpg', '.txt'), 'r') as f:
        lines = f.readlines()
    bboxes = []
    classes = []
    for line in lines:
        line = line.strip().split(' ')
        x_center, y_center, w, h = map(float, line[1:])
        # if int(line[0]) != class_id:
        #     continue
        x_min = float(x_center - w / 2) * img.shape[1]
        y_min = float(y_center - h / 2) * img.shape[0]
        x_max = float(x_center + w / 2) * img.shape[1]
        y_max = float(y_center + h / 2) * img.shape[0]
        # Crop the image using slicing 
        # print(x_min, y_min, x_max, y_max)
        # print('====++===================')
        bboxes.append((x_min, y_min, x_max, y_max))
        classes.append(int(line[0]))
    return bboxes, classes

def infer_bbox(img_path, model, class_id):
    
    # img_path = '/home/mq/data_disk2T/Thang/bak/src/data1/train/images/247492b253c1630674da3651bdc1259b.jpg'
    image = cv2.imread(img_path)
    image_height, image_width = image.shape[0], image.shape[1]
    result = model(img_path, verbose = False)
    bboxes_infer = []
    classes_infer = []
    all_img_infer = []
    for i in range(len(result[0].boxes.xywh)):
        if int(result[0].boxes.cls[i].cpu().numpy()) != class_id:
            continue
        x, y, w, h = result[0].boxes.xywh[i].cpu().numpy()
        x_min = float(x - w / 2) 
        y_min = float(y - h / 2) 
        x_max = float(x + w / 2) 
        y_max = float(y + h / 2)
        # print(x_min, y_min, x_max, y_max)
        # print('-------------------')
        bboxes_infer.append((x_min, y_min, x_max, y_max))
        classes_infer.append(int(result[0].boxes.cls[i].cpu().numpy()))
        cropped = torch.from_numpy(image[int(y_min):int(y_max), int(x_min):int(x_max)])
        all_img_infer.append(cropped[None, ...].permute(0, 3, 1, 2))
    return all_img_infer, result[0].boxes.xywh, image, bboxes_infer, classes_infer
    # return bboxes_infer, classes_infer

def calculate_iou(box1, box2):
    """
    Calculate the Intersection Over Union (IOU) of two bounding boxes.
    
    Args:
    - box1: Tuple (x1, y1, x2, y2) for the first bounding box.
    - box2: Tuple (x1, y1, x2, y2) for the second bounding box.
    
    Returns:
    - iou: Intersection Over Union value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Compute intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0
    
    iou = intersection_area / union_area
    return iou

def benchmark_outlier_detection(pred_bboxes, gt_bboxes, pred_classes, gt_classes, outliers, iou_threshold=0.4):
    """
    Benchmark outlier detection for object detection.
    
    Args:
    - pred_bboxes (list): List of predicted bounding boxes [(x1, y1, x2, y2), ...].
    - gt_bboxes (list): List of ground truth bounding boxes [(x1, y1, x2, y2), ...].
    - pred_classes (list): List of predicted classes corresponding to pred_bboxes.
    - gt_classes (list): List of ground truth classes corresponding to gt_bboxes.
    - iou_threshold (float): IOU threshold for matching (default: 0.7).
    
    Returns:
    - sample_space (int): Total points based on the specified conditions.
    """
    outlier_positive = 0
    inlier_positive = 0
    outlier_sample = 0
    matched_gt = set()  # Keep track of matched ground truth boxes
    for i, pred_box in enumerate(pred_bboxes):
        pred_class = pred_classes[i]
        max_iou = 0
        best_match_idx = -1

        # Find the ground truth box with the maximum IOU
        for j, gt_box in enumerate(gt_bboxes):
            # if j in matched_gt:  # Skip already matched ground truth boxes
            #     continue
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                best_match_idx = j
        if best_match_idx == -1:  # No intersection
            outlier_sample += 1
            if outliers[i] == 1:
                outlier_positive += 1 # Add a point for no intersection

        else:
            gt_class = gt_classes[best_match_idx]
            # matched_gt.add(best_match_idx)  # Mark as matched
            # print(max_iou)
            
            # Apply conditions based on IOU and class
            if max_iou >= iou_threshold:
                # matched_gt.add(best_match_idx)
     
                if pred_class == gt_class:
                    if outliers[i] == 0:
                        inlier_positive += 1
                #     pass  # Add a point for high IOU and same class
                else:
                    outlier_sample += 1
                    if outliers[i]==1:
                        outlier_positive += 1
                #     pass # Add a point for high IOU and different class
            else:
                outlier_sample += 1
                if outliers[i] == 1:
                    outlier_positive += 1
                pass  # Do not add a point for low IOU

    return inlier_positive, outlier_positive, outlier_sample

if __name__ == '__main__':
    clf = LUNAR(verbose = 1, negative_sampling='UNIFORM', n_epochs=300, n_neighbours = 2, contamination = 0.3)
    model_detect = YOLO('/home/mq/data_disk2T/Thang/ps/weights/best.pt').cuda()
    model_feat = init_model()
    image_folder = glob.glob('/home/mq/data_disk2T/HoangChu/JSON2YOLO/gt_dir/images/train/*.jpg')
    total_sample = 0
    total_inlier_positive = 0
    total_outlier_positive = 0
    clf = load('clf_erd.joblib')
    count = 0
    for img_path in image_folder:
        count += 1
        if count > 400: break
        
        if not os.path.exists(img_path.replace('images/', 'labels/').replace('.jpg', '.txt')):
            continue
        gt_bboxes, gt_classes = read_txt_bbox(img_path, class_id=0)
        crop_bboxes, bboxes, image, pred_bboxes, pred_classes = infer_bbox(img_path, model_detect, class_id=0)
        total_sample += len(pred_bboxes)
        feats_each_img = extract_feature(crop_bboxes, model_feat)
        # print('**********************')
        if len(feats_each_img) == 0:
                continue
        outliers = clf.predict(feats_each_img.squeeze(1).numpy())
        inlier_positive, outlier_positive, outlier_sample = benchmark_outlier_detection(pred_bboxes, gt_bboxes, pred_classes, gt_classes, outliers, iou_threshold=0.7)
        total_inlier_positive += inlier_positive
        total_outlier_positive += outlier_positive
        total_sample += outlier_sample
    # recall
    print((total_outlier_positive)/total_sample)
    # Wrong Inliers 