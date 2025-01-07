import os
import cv2
import glob
import torch
import numpy as np
from test import visualize
from ultralytics import YOLO
from joblib import dump, load
from torch.nn import functional as F
from ood.pyod.pyod.models.vae import VAE
from ood.pyod.pyod.models.lunar import LUNAR
from ood.pyod.pyod.models.lof import LOF
from ood.pyod.pyod.models.copod import COPOD
from ood.pyod.pyod.models.iforest import IForest
from ood.pyod.pyod.models.thresholds import FILTER, FWFM
from ood.pyod.pyod.models.suod import SUOD

def pad_image_to_size(image, target_size=(100, 100)):
    """
    Pad an image tensor to ensure its dimensions are at least target_size.
    
    Args:
    - image (Tensor): The image tensor with shape (C, H, W).
    - target_size (tuple): The target size as (height, width).
    
    Returns:
    - Padded image tensor.
    """
    _, _, h, w = image.shape
    target_h, target_w = target_size

    # Compute padding
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    # Pad (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)  # (W_before, W_after, H_before, H_after)
    
    # Apply padding
    padded_image = F.pad(image.squeeze(0), padding, mode="constant", value=0)  # Padding with 0 (black)
    
    return padded_image[None,...]


def read_txt_bbox(img_path, class_id):

    img = cv2.imread(img_path)
    with open(img_path.replace('images/', 'labels/').replace('.jpg', '.txt'), 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        line = line.strip().split(' ')
        x_center, y_center, w, h = map(float, line[1:])
        if int(line[0]) != class_id:
            continue
        x_min = float(x_center - w / 2) * img.shape[1]
        y_min = float(y_center - h / 2) * img.shape[0]
        x_max = float(x_center + w / 2) * img.shape[1]
        y_max = float(y_center + h / 2) * img.shape[0]
        # Crop the image using slicing 
        cropped = torch.from_numpy(img[int(y_min):int(y_max), int(x_min):int(x_max)])
        bboxes.append(cropped[None, ...].permute(0, 3, 1, 2))
    return bboxes

def init_model():

    model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')
    return model

def display_ood_data(pred, img_path, image, class_id, bboxes):
    bboxes_pred_ood = []
    bboxes_pred_id = []
    for idx, value in enumerate(pred):
        if value == 1:
            bboxes_pred_ood.append(bboxes[idx])
        else:
            bboxes_pred_id.append(bboxes[idx])
    out_path = '/home/mq/data_disk2T/Thang/ps/al/TEST/pred/{}'.format(img_path.split('/')[-1])
    visualize(bboxes_pred_ood, image, class_id, out_path, color = (0, 0, 255))
    visualize(bboxes_pred_id, image, class_id, out_path, color = (255, 0, 0))


def extract_feature(bboxes, model):
    model.eval()
    feats_each_img = []  # Use a list to collect tensors
    for cropped in bboxes:
        if cropped.size(2) < 100 or cropped.size(3) < 100:
            # padding images into size 100, 100
            cropped = pad_image_to_size(cropped)
        with torch.no_grad():  # Disable gradient computation
            feats = model(cropped.float())
            feats_each_img.append(feats.cpu().numpy())  # Append tensor to the list
    # Concatenate all features along dimension 0
    return torch.tensor(feats_each_img)


def infer_cropped_image(img_path, model, class_id):
    
    # img_path = '/home/mq/data_disk2T/Thang/bak/src/data1/train/images/247492b253c1630674da3651bdc1259b.jpg'
    image = cv2.imread(img_path)
    image_height, image_width = image.shape[0], image.shape[1]
    result = model(img_path, verbose = False)
    all_img_infer = []
    boxes = []
    for i in range(len(result[0].boxes.xywh)):
        if int(result[0].boxes.cls[i].cpu().numpy()) != class_id:
            continue
        x, y, w, h = result[0].boxes.xywh[i].cpu().numpy()
        x_min = float(x - w / 2) 
        y_min = float(y - h / 2) 
        x_max = float(x + w / 2) 
        y_max = float(y + h / 2)
        cropped = torch.from_numpy(image[int(y_min):int(y_max), int(x_min):int(x_max)])
        boxes.append((x, y, w, h))
        all_img_infer.append(cropped[None, ...].permute(0, 3, 1, 2))
    return all_img_infer, boxes, image
        # list_cls_result = [result[0].boxes.cls[i].cpu().numpy() for i in range(len(result[0].boxes.cls))]

if __name__ == '__main__':
    model_feat = init_model()
    model_detect = YOLO('/home/mq/data_disk2T/Thang/ps/weights/best.pt').cuda()
    # img_folder = glob.glob('/home/mq/data_disk2T/HoangChu/JSON2YOLO/gt_dir/images/train/*.jpg')
    img_folder = glob.glob('/home/mq/data_disk2T/Thang/ps/test_folder/*.jpg')
    # clf = LUNAR(verbose = 1, negative_sampling='UNIFORM', n_epochs=300, n_neighbours = 5, contamination = 0.5)
    # initialized a group of outlier detectors for acceleration
    detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                    LOF(n_neighbors=25), LOF(n_neighbors=35),
                    COPOD(), IForest(n_estimators=100),
                    IForest(n_estimators=200)]

    # decide the number of parallel process, and the combination method
    # then clf can be used as any outlier detection model
    clf = SUOD(base_estimators=detector_list, n_jobs=1, combination='average',
            verbose=True, contamination=0.5)
    clf = load('clf_test.joblib')
    all_feats = torch.tensor([])
    count = 0
    for img_path in img_folder:
        count += 1
        if count > 400: break
        # if not os.path.exists(img_path.replace('images/', 'labels/').replace('.jpg', '.txt')):
        #     print('pass')
        #     continue
        # crop_bboxes = read_txt_bbox(img_path, class_id=0)
        crop_bboxes, bboxes, image = infer_cropped_image(img_path, model_detect, class_id=0)
        feats_each_img = extract_feature(crop_bboxes, model_feat)
        if len(feats_each_img) == 0:
            continue
        # all_feats = torch.cat((all_feats, feats_each_img), dim=0)
        ood = clf.predict(feats_each_img.squeeze(1).numpy())
        print(ood)
        # print(clf.decision_function(feats_each_img.squeeze(1).numpy()))
        display_ood_data(ood, img_path, image, class_id = 0, bboxes = bboxes)


    # print(all_feats.squeeze(1).shape)
    # clf.fit(all_feats.squeeze(1).numpy())
    # dump(clf, 'clf_test.joblib')    
   
    
        



