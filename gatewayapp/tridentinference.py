# Copyright (C) 2020 - 2022 APC, Inc.

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
# setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from TridentNet.tridentnet import add_tridentnet_config

predictor = None
cfg = None

def setupPredictor():
    global predictor
    global cfg
    if predictor is None:
        print("setup predictor")
        cfg = get_cfg()
        add_tridentnet_config(cfg)
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file('./TridentNet/configs/tridentnet_fast_R_101_C4_3x.yaml')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = './TridentNet/model/model_final_164568.pkl' #TridentFast	R101-C4	C5-128RO
        predictor = DefaultPredictor(cfg)

def inference(im):
    outputs = predictor(im)    
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)

    v3 = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))

    pred_boxes = outputs["instances"].pred_boxes.to('cpu')
    pred_scores = outputs["instances"].scores.to('cpu')
    pred_classes = outputs["instances"].pred_classes.to('cpu')
    labels = v3.metadata.get("thing_classes", None)

    result = {}
    result["numberOfObjects"] = len(pred_boxes)
    objects = []
    for box, score, _class in zip (pred_boxes, pred_scores, pred_classes):
        print("box", box.numpy(), "score", int(score.numpy()*100), "class_id", _class.numpy(), "label", labels[_class.numpy()])
        objects.append({"box": box.numpy().tolist(), "score": int(score.numpy()*100), "label": labels[_class.numpy()]})

    #     v3.draw_box(box)
    #     v3.draw_text(labels[_class.numpy()] + " " + str(int(score.numpy()*100)) + "%", tuple(box[:2].numpy()))
    # v3 = v3.get_output()    
    # img_output =  v3.get_image()[:, :, ::-1]
    # # cv2.imwrite("output.png", img_output)
    # return img_output
    result["objects"] = objects
    return json.dumps(result)
    

if __name__ == '__main__':
    setupPredictor()
    im = cv2.imread("./input.jpg")
    print("im shape", im.shape)
    inference(im)
        
