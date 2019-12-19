# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_val", {}, "/media/peter/268cfab1-9277-4a15-9d03-aaa2d1547bd0/peter/frist_year_up/basic_on/HW4/HW4/detectron2_repo/dataset_HW4/test.json", "/media/peter/268cfab1-9277-4a15-9d03-aaa2d1547bd0/peter/frist_year_up/basic_on/HW4/HW4/detectron2_repo/dataset_HW4/test_images")
#detectron2.data.datasets.load_coco_json("/media/peter/268cfab1-9277-4a15-9d03-aaa2d1547bd0/peter/frist_year_up/basic_on/HW4/HW4/detectron2_repo/dataset_HW4/pascal_train.json", "/media/peter/268cfab1-9277-4a15-9d03-aaa2d1547bd0/peter/frist_year_up/basic_on/HW4/HW4/detectron2_repo/dataset_HW4/train_images", dataset_name=None, extra_annotation_keys=None)

#start to train model
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ()
cfg.DATASETS.TEST = ("my_dataset_val", )
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # only has one class (ballon)

predictor = DefaultPredictor(cfg)# run inference of your model

from detectron2.utils.visualizer import ColorMode
cocoGt = COCO("./dataset_HW4/test.json")
dataset_dicts = "./dataset_HW4/test.json"
balloon_metadata = MetadataCatalog.get("my_dataset_val")

from utils_HW4 import binary_mask_to_rle

coco_dt = []

for imgid in cocoGt.imgs:    
    im = cv2.imread("./dataset_HW4/test_images/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1] # load image
    
    outputs = predictor(im)
    # v = Visualizer(im[:, :, ::-1],
    #                metadata=balloon_metadata, 
    #                scale=0.8, 
    #                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    # )
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow("", v.get_image()[:, :, ::-1])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
   

    # a=(outputs['instances'].to("cpu")).scores
    # a=a.tolist()
    # if a:
    #   print(a[0])
    # else:
    #   print(0)
    n_instances = len((outputs['instances'].to("cpu")).scores)

    if len((outputs['instances'].to("cpu")).pred_classes) > 0: # If any objects are detected in this image
        for i in range(n_instances): # Loop all instances
            # save information of the instance in a dictionary then append on coco_dt list
            pred = {}
            pred['image_id'] = imgid # this imgid must be same as the key of test.json

            outputs_pred_classes=(outputs['instances'].to("cpu")).pred_classes
            outputs_pred_classes=outputs_pred_classes.tolist()
            pred['category_id'] = int(outputs_pred_classes[i]+1)
            # tmp = (outputs['instances'].to("cpu")).pred_masks[i,:,:]
            # print (tmp.size())
            # print (stop)


            pred['segmentation'] = binary_mask_to_rle(((outputs['instances'].to("cpu")).pred_masks[i,:,:]).numpy()) # save binary mask to RLE, e.g. 512x512 -> rle
            #print(type((outputs['instances'].to("cpu")).pred_masks[:,:,i]) )
            
            outputs_scores=(outputs['instances'].to("cpu")).scores
            outputs_scores=outputs_scores.tolist()
            pred['score'] = float(outputs_scores[i])



            coco_dt.append(pred)
            

with open("submission.json", "w") as f:
    json.dump(coco_dt, f)

       