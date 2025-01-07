from plugins.extract import Feats_Extraction
from ultralytics import YOLO
import glob
import cv2
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import yaml_load
from ultralytics.nn.tasks import yaml_model_load, parse_model
from copy import deepcopy
import numpy as np
from ood.pyod.pyod.models.lunar import LUNAR
from ood.pyod.pyod.models.vae import VAE
from joblib import dump, load
from test import visualize, read_txt

json_cfg = {
        "trainset": '/home/mq/data_disk2T/Thang/bak/src/data1/data1.yaml',
        "quality": "json/quality.json",
        "uncertainty": "json/uncertainty.json",
        "feat": 'json/new.npy',
        "uncertainty_out": 'json/out.txt'
    }
class_id = 0
img_folder = glob.glob('/home/mq/data_disk2T/Thang/bak/src/data1/train/images/*.jpg')
clf = VAE()
f = open('/home/mq/data_disk2T/Thang/ps/al/feats_distribution.npy', 'rb')
X_train = np.load(f)
clf.fit(X_train)
dump(clf, 'clf.joblib')
 # load the model
clf = load('clf.joblib')
for img_path in img_folder:
    # img_path = '/home/mq/data_disk2T/Thang/bak/src/data1/train/images/247492b253c1630674da3651bdc1259b.jpg'
    image = np.array(cv2.imread(img_path))
    new_image = np.array(cv2.imread(img_path))
    image_height, image_width = image.shape[0], image.shape[1]
    model = YOLO('/home/mq/data_disk2T/Thang/bak/src/runs/detect/train10/weights/best.pt').cuda()
    result = model(img_path)
    list_result = [result[0].boxes.xywhn[i].cpu().numpy() for i in range(len(result[0].boxes.xywhn))]
    list_cls_result = [result[0].boxes.cls[i].cpu().numpy() for i in range(len(result[0].boxes.cls))]
    feats_extraction = Feats_Extraction(model, json_cfg['uncertainty_out'], '/home/mq/data_disk2T/Thang/ps/al/ultralytics/cfg/models/v8/yolov8.yaml')
    det_labels, det_bboxes, det_feats, det_scores, anchor_points, pred_feats = feats_extraction.extract_feature(img_path)
    all_feats = []
    new_list_result = []
    for idex, k in enumerate(list_result):
        if int(list_cls_result[idex]) != class_id:
            continue
        x_center = k[0]
        y_center = k[1]   
        points = np.array([[x_center*20, y_center*20],[x_center*40, y_center*40],[x_center*80, y_center*80]])
        distances = np.sqrt(((points[:, None] - anchor_points.cpu().numpy()) ** 2).sum(axis=2))

        # Find the index of the nearest anchor for each point
        nearest_indices = np.argmin(distances, axis=1)

        merge_feats = pred_feats[0,nearest_indices,...].flatten(0).cpu().numpy()
        new_list_result.append(k)
        all_feats.append(merge_feats)
    if all_feats == []:
        continue
    all_feats = np.array(all_feats)

    pred = clf.predict(all_feats)
    y_test_scores = clf.decision_function(all_feats)
    bboxes_pred_ood = []
    bboxes_pred_id = []
    for idx, value in enumerate(pred):
        if value == 1:
            bboxes_pred_ood.append(new_list_result[idx])
        else:
            bboxes_pred_id.append(new_list_result[idx])
    out_path = '/home/mq/data_disk2T/Thang/ps/al/TEST/pred/{}'.format(img_path.split('/')[-1])
    visualize(bboxes_pred_ood, image_width, image_height, image, class_id, out_path, color = (0, 0, 255))
    visualize(bboxes_pred_id, image_width, image_height, image, class_id, out_path, color = (255, 0, 0))

    out_gt_path = '/home/mq/data_disk2T/Thang/ps/al/TEST/gt/{}'.format(img_path.split('/')[-1])

    
    gt_bboxes = read_txt(img_path.replace('images', 'labels').replace('.jpg', '.txt'))
    visualize(gt_bboxes, image_width, image_height, new_image, class_id, out_gt_path)





