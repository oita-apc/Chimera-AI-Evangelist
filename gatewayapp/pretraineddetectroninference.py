# Copyright (C) 2020 - 2022 APC, Inc.

# Some basic setup:
# from detectron2.utils.logger import setup_logger
# setup_logger()

# import some common libraries
import numpy as np
import json, cv2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.utils.visualizer import GenericMask

from detectron2.projects.point_rend import add_pointrend_config
from TridentNet.tridentnet import add_tridentnet_config

predictor_mask_rcnn = None
predictor_pointrend = None
predictor_trident = None
predictor_faster_rcnn = None
metadata_catalog = None

def setupPredictor(model_type):
    global predictor_mask_rcnn, metadata_catalog, predictor_pointrend
    global predictor_trident, predictor_faster_rcnn
    global cfg
    if model_type == 'mask_rcnn' and predictor_mask_rcnn is None:
        print("setup predictor mask r-cnn")
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.merge_from_file('./maskrcnn/configs/mask_rcnn_R_50_FPN_3x.yaml')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = './maskrcnn/model/model_final_f10217.pkl' #mask_rcnn_R_50_FPN_3x
        predictor_mask_rcnn = DefaultPredictor(cfg)
        metadata_catalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    elif model_type == 'pointrend' and predictor_pointrend is None:
        print("setup predictor pointrend")
        cfg = get_cfg()
        add_pointrend_config(cfg)
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.merge_from_file('./pointrend/configs/pointrend_rcnn_R_50_FPN_3x_coco.yaml')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = './pointrend/model/model_final_edd263.pkl' #pointrend R50-FPN-3x
        predictor_pointrend = DefaultPredictor(cfg)
        metadata_catalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    elif model_type == 'trident' and predictor_trident is None:
        print("setup predictor trident")
        cfg = get_cfg()
        add_tridentnet_config(cfg)
        cfg.merge_from_file('./TridentNet/configs/tridentnet_fast_R_101_C4_3x.yaml')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = './TridentNet/model/model_final_164568.pkl' #TridentFast	R101-C4	C5-128RO
        predictor_trident = DefaultPredictor(cfg)
        metadata_catalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    elif model_type == 'faster_rcnn' and predictor_faster_rcnn is None:
        print("setup predictor faster r-cnn")
        cfg = get_cfg()
        cfg.merge_from_file("./fasterrcnn/configs/faster_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = './fasterrcnn/model/faster_rcnn_R_101_FPN_3x_model_final.pkl' 
        predictor_faster_rcnn = DefaultPredictor(cfg)
        metadata_catalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

def inference(im, model_type):
    setupPredictor(model_type)
    if model_type == 'mask_rcnn':
        outputs = predictor_mask_rcnn(im)
    elif model_type == 'pointrend':
        outputs = predictor_pointrend(im)
    elif model_type == 'trident':
        outputs = predictor_trident(im)
    elif model_type == 'faster_rcnn':
        outputs = predictor_faster_rcnn(im)
    
    result = {}
    result["numberOfObjects"] = 0
    result["objects"] = []
    result["modelType"] = model_type

    if ("outputs" in locals()): # check if outputs variable exist 
        # print(outputs["instances"].pred_classes)
        # print(outputs["instances"].pred_boxes)

        v3 = Visualizer(im[:, :, ::-1], metadata_catalog)

        pred_boxes = outputs["instances"].pred_boxes.to('cpu')
        pred_scores = outputs["instances"].scores.to('cpu')
        pred_classes = outputs["instances"].pred_classes.to('cpu')
        labels = v3.metadata.get("thing_classes", None)
        print("labels", labels)
        
        objects = []

        if model_type == 'mask_rcnn' or model_type == 'pointrend':
            result["numberOfObjects"] = len(pred_boxes)
            # pred_masks = outputs["instances"].pred_masks.to('cpu')
            pred_masks = np.asarray(outputs["instances"].pred_masks.to('cpu'))
            # print("masks", pred_masks.shape)                                
            for box, score, _class, mask in zip (pred_boxes, pred_scores, pred_classes, pred_masks):
                print("box", box.numpy(), "score", int(score.numpy()*100), "class_id", _class.numpy(), "label", labels[_class.numpy()], "mask", mask.shape)
                genmask = GenericMask(mask, im.shape[0], im.shape[1])
                if len(genmask.polygons) == 0:
                    continue
                polygon = genmask.polygons[0].reshape(-1,2)
                print("polygon shape", polygon.shape)
                objects.append({"box": box.numpy().tolist(), "polygon": polygon.tolist(), "score": int(score.numpy()*100), "label": labels[_class.numpy()]})
        elif model_type == 'faster_rcnn' or model_type == 'trident':
            result["numberOfObjects"] = len(pred_boxes)                        
            for box, score, _class in zip (pred_boxes, pred_scores, pred_classes):
                print("box", box.numpy(), "score", int(score.numpy()*100), "class_id", _class.numpy(), "label", labels[_class.numpy()])
                objects.append({"box": box.numpy().tolist(), "score": int(score.numpy()*100), "label": labels[_class.numpy()]})

        result["objects"] = objects

    return json.dumps(result)
    

if __name__ == '__main__':
    # model_type = 'mask_rcnn'
    # model_type = 'pointrend'
    # model_type = 'trident'
    model_type = 'faster_rcnn'
    # setupPredictor(model_type)
    im = cv2.imread("./input.jpg")
    print("im shape", im.shape)
    ret = inference(im, model_type)
    print(ret)
        
