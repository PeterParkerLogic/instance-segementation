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


from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "/media/peter/268cfab1-9277-4a15-9d03-aaa2d1547bd0/peter/frist_year_up/basic_on/HW4/HW4/detectron2_repo/dataset_HW4/pascal_train.json", "/media/peter/268cfab1-9277-4a15-9d03-aaa2d1547bd0/peter/frist_year_up/basic_on/HW4/HW4/detectron2_repo/dataset_HW4/train_images")
#detectron2.data.datasets.load_coco_json("/media/peter/268cfab1-9277-4a15-9d03-aaa2d1547bd0/peter/frist_year_up/basic_on/HW4/HW4/detectron2_repo/dataset_HW4/pascal_train.json", "/media/peter/268cfab1-9277-4a15-9d03-aaa2d1547bd0/peter/frist_year_up/basic_on/HW4/HW4/detectron2_repo/dataset_HW4/train_images", dataset_name=None, extra_annotation_keys=None)

#start to train model
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "./output_10000/model_final.pth"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 15000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
