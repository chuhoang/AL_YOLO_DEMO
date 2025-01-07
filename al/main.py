from plugins.quality import Quality
from plugins.uncertainty import Uncertainty
from plugins.difficulty import Calibrated_Sampler
from plugins.extract import Feats_Extraction
from plugins.disversity import DiversitySampler
from ultralytics import YOLO
import glob
import cv2
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import yaml_load
from ultralytics.nn.tasks import yaml_model_load, parse_model
from copy import deepcopy
import numpy as np
if __name__ == '__main__':

    model = YOLO('/home/mq/data_disk2T/Thang/bak/src/runs/detect/train10/weights/best.pt').cuda()
  
    # data = yaml_load('/home/mq/data_disk2T/Thang/bak/src/data1/data.yaml')
    # dataset = YOLODataset(img_path = '/home/mq/data_disk2T/Thang/bak/src/data1/train/images',data= data, task='detect')

    # for b in dataset:
    #     print(b)
    #read image , resize 640x640, transpose
    # img = cv2.imread('../weights/bus.jpg')
    # img = cv2.resize(img, (640, 640))
    # img = img.transpose(2, 0, 1)/255
    # img = img[None,...]

    # x = torch.from_numpy(img).float()

    # r = model(x)
    # print(r[1][2].shape)

    ls = glob.glob('/home/mq/data_disk2T/Thang/bak/src/data1/train/images/*.jpg')

    json_cfg = {
        "trainset": '/home/mq/data_disk2T/Thang/bak/src/data1/data1.yaml',
        "quality": "json/quality.json",
        "uncertainty": "json/uncertainty.json",
        "feat": 'json/new.npy',
        "uncertainty_out": 'json/out.txt'
    }

    # # # # preprocess
    # quality = Quality(model, base_momentum=0.99, img_path='/home/mq/data_disk2T/Thang/bak/src/data1/train/images')
    # quality(trainset=json_cfg['trainset'] , out=json_cfg['quality'])

    # # # # plugins
    # uncertainty = Uncertainty(model)
    # uncertainty(ls, out=json_cfg['uncertainty'])
    
    # calibrated_sampler = Calibrated_Sampler(n_images= int(0.3*len(ls)), score_thr=0.01, floor_score_thr=0.6, category_valid=[0,2,80,81,82,83,84])
    # idxs_valid, idxs_non_valid = calibrated_sampler(json_cfg['quality'], json_cfg['uncertainty'], max_entropy_filter=True)
    # # idxs_valid = [2, 2, 2, 338, 120, 377, 358, 92, 57, 13, 339, 194, 402, 35, 238, 487, 165, 66, 263, 190, 86, 532, 280, 221, 359, 269, 423, 279, 179, 276]
    # idxs_valid = list(range(len(ls)))
    feats_extraction = Feats_Extraction(model, json_cfg['uncertainty_out'], '/home/mq/data_disk2T/Thang/ps/al/ultralytics/cfg/models/v8/yolov8.yaml')
    all_feats = []
    for img_path in ls:    
        # img = cv2.imread('/home/mq/data_disk2T/Thang/bak/src/data1/val/images/0ac4918462e96b54cac4ecaa0f3056fa.jpg')
        det_labels, det_bboxes, det_feats, det_scores, anchor_points, pred_feats = feats_extraction.extract_feature(img_path)
        
        f = open(img_path.replace('images', 'labels').replace('.jpg', '.txt'), 'r')
        a = f.readlines()
        for i in a:
            k = i.strip().split(' ')
            k = [float(j) for j in k]
            if k[0] == 0:
            # x_center = k[1]+(k[3]/2)
            # y_center = k[2]+(k[4]/2)   
                x_center = k[1]
                y_center = k[2] 
                points = np.array([[x_center*20, y_center*20],[x_center*40, y_center*40],[x_center*80, y_center*80]])
                distances = np.sqrt(((points[:, None] - anchor_points.cpu().numpy()) ** 2).sum(axis=2))

                # Find the index of the nearest anchor for each point
                nearest_indices = np.argmin(distances, axis=1)
                merge_feats = pred_feats[0,nearest_indices,...].flatten(0).cpu().numpy()
                all_feats.append(merge_feats)
                # with open('feats_distribution.npy', 'wb') as fwb:
                #         np.save(fwb, merge_feats)
    all_feats = np.array(all_feats)
    with open('feats_distribution.npy', 'wb') as fwb:
            np.save(fwb, all_feats)
        # print(pred_feats.shape)
       
    #     exit()
    # feats_extraction(idxs_valid, ls, out=json_cfg['feat'])

    # diversity_sampler = DiversitySampler(K=int(0.3*int(0.3*len(ls))))
    # sampled_path, unsampled_path = diversity_sampler(feature_file=json_cfg['feat'], uncertainty_file=json_cfg['uncertainty_out'])
    
    # print(len(sampled_path), len(unsampled_path))
    # print(sampled_path)
    