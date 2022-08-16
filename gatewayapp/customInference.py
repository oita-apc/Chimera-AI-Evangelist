# Copyright (C) 2020 - 2022 APC, Inc.

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
# setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, sys

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from TridentNet.tridentnet import add_tridentnet_config
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.utils.visualizer import GenericMask
import json

sys.path.insert(1, os.path.join(sys.path[0], './detr'))
from detr.d2.detr import add_detr_config

from utility import get_base_dir

g_predictor = None
g_cfg = None
g_labels = None
g_modelType = ''

def detr_filter_predictions_from_outputs(outputs,
                                    threshold=0.7,
                                    verbose=False):

  predictions = outputs["instances"].to("cpu")

  if verbose:
    print(list(predictions.get_fields()))

  # Reference: https://github.com/facebookresearch/detectron2/blob/7f06f5383421b847d299b8edf480a71e2af66e63/detectron2/structures/instances.py#L27
  #
  #   Indexing: ``instances[indices]`` will apply the indexing on all the fields
  #   and returns a new :class:`Instances`.
  #   Typically, ``indices`` is a integer vector of indices,
  #   or a binary mask of length ``num_instances``

  indices = [i
            for (i, s) in enumerate(predictions.scores)
            if s >= threshold
            ]

  filtered_predictions = predictions[indices]

  return filtered_predictions

def setupPredictor(saved_model_path, modelType, model_weigth_file=None):
    global g_predictor, g_modelType
    global g_cfg
    global g_labels
    # g_modelType = str(np.loadtxt(MODEL_TYPE_PATH, dtype=str))
    g_modelType = modelType
    if g_predictor is None:
        print("try setup custom predictor", g_modelType)
        if (g_modelType == 'trident' or g_modelType == 'detr' or g_modelType == 'faster_rcnn'
            or g_modelType == 'mask_rcnn' or g_modelType == 'pointrend'):
            g_cfg = get_cfg()
            if (g_modelType == 'trident'):
                add_tridentnet_config(g_cfg)
                # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
                g_cfg.merge_from_file('./TridentNet/configs/tridentnet_fast_R_101_C4_3x.yaml')
                g_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
            elif (g_modelType == 'detr'):
                add_detr_config(g_cfg)
                g_cfg.merge_from_file('./detr/d2/configs/detr_256_6_6_torchvision.yaml')
                g_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
            elif (g_modelType == 'faster_rcnn'):                
                g_cfg.merge_from_file("./fasterrcnn/configs/faster_rcnn_R_101_FPN_3x.yaml")
                g_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
            elif (g_modelType == 'mask_rcnn'):
                g_cfg.merge_from_file('./maskrcnn/configs/mask_rcnn_R_50_FPN_3x.yaml')
                g_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            elif (g_modelType == 'pointrend'):
                add_pointrend_config(g_cfg)
                g_cfg.merge_from_file('./pointrend/configs/pointrend_rcnn_R_50_FPN_3x_coco.yaml')
                g_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            g_cfg.MODEL.WEIGHTS = os.path.join(saved_model_path, "model." + g_modelType)
            if (model_weigth_file is not None):
                g_cfg.MODEL.WEIGHTS = model_weigth_file
            print("weight", g_cfg.MODEL.WEIGHTS)
            if (os.path.isfile(g_cfg.MODEL.WEIGHTS)):            
                print("did setup custom predictor")
                # TODO classes.txt should contain only labelId not labelName
                g_labels = np.loadtxt(os.path.join(saved_model_path, 'classes.txt'), dtype=str)
                g_cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(g_labels)
                if (g_modelType == 'detr'):
                    g_cfg.MODEL.DETR.NUM_CLASSES = g_cfg.MODEL.ROI_HEADS.NUM_CLASSES
                elif (g_modelType == 'pointrend'):
                    g_cfg.MODEL.POINT_HEAD.NUM_CLASSES = g_cfg.MODEL.ROI_HEADS.NUM_CLASSES
                g_predictor = DefaultPredictor(g_cfg)
            else:
                print(g_cfg.MODEL.WEIGHTS, " is not exist!")
                

def inference(im, draw=False):
    global g_predictor
    if (g_predictor is None):
        print("predictor is not ready yet!!")
        return '{"numberOfObjects":0, "objects":[]}'
    print("try to predict with custom model ", g_modelType)
    outputs = g_predictor(im)    
        
    if (draw):
        v3 = Visualizer(im[:, :, ::-1])
    
    if g_modelType == 'detr':
        outputs = detr_filter_predictions_from_outputs(outputs, threshold=g_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
        pred_boxes = outputs.pred_boxes
        pred_scores = outputs.scores
        pred_classes = outputs.pred_classes
    else:
        pred_boxes = outputs["instances"].pred_boxes.to('cpu')
        pred_scores = outputs["instances"].scores.to('cpu')
        pred_classes = outputs["instances"].pred_classes.to('cpu')

    print("pred_classes",pred_classes)
    print("pred_boxes",pred_boxes)
    print("pred_scores",pred_scores)

    result = {}    
    objects = []
    if g_modelType == 'detr' or g_modelType == 'faster_rcnn' or g_modelType == 'trident':
        for box, score, _class in zip (pred_boxes, pred_scores, pred_classes):
            if (score < g_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST):
                continue
            print("box", box.numpy(), "score", int(score.numpy()*100), "class_id", _class.numpy(), "label", g_labels[_class.numpy()])
            objects.append({"box": box.numpy().tolist(), "score": int(score.numpy()*100), "label": g_labels[_class.numpy()]})

            if (draw):
                v3.draw_box(box)        
                v3.draw_text("label:" + str(_class.numpy()) + " " + str(int(score.numpy()*100)) + "%", tuple(box[:2].numpy()))
    elif g_modelType == 'mask_rcnn' or g_modelType == 'pointrend':
        # pred_masks = outputs["instances"].pred_masks.to('cpu')
        pred_masks = np.asarray(outputs["instances"].pred_masks.to('cpu'))
        print("masks", pred_masks.shape)                                
        for box, score, _class, mask in zip (pred_boxes, pred_scores, pred_classes, pred_masks):
            print("box", box.numpy(), "score", int(score.numpy()*100), "class_id", _class.numpy(), "label", g_labels[_class.numpy()], "mask", mask.shape)
            genmask = GenericMask(mask, im.shape[0], im.shape[1])
            if len(genmask.polygons) == 0:
                continue
            polygon = genmask.polygons[0].reshape(-1,2)
            print("polygon shape", polygon.shape)
            objects.append({"box": box.numpy().tolist(), "polygon": polygon.tolist(), "score": int(score.numpy()*100), "label": g_labels[_class.numpy()]})

            if (draw):
                v3.draw_instance_predictions(outputs["instances"].to('cpu'))
                # v3.draw_box(box)        
                # v3.draw_text("label:" + str(_class.numpy()) + " " + str(int(score.numpy()*100)) + "%", tuple(box[:2].numpy()))
    
    if (draw):
        v3 = v3.get_output()    
        img_output =  v3.get_image()[:, :, ::-1]
        cv2.imwrite("output.png", img_output)
        print("save output.png")

    result["objects"] = objects
    result["numberOfObjects"] = len(objects)
    result["modelType"] = g_modelType
    resultStr = json.dumps(result)
    print("resultStr", resultStr)
    return resultStr
    

if __name__ == '__main__':
    import os
    # working_directory = '/workspace-test-v1/mlapp/working_directory'
    # saved_model_path =  '/workspace-test-v1/saved-model'
    working_directory = '/data/workspace-demoapp-ai-qa1/mlapp/working_directory'
    saved_model_path = '/data/workspace-demoapp-ai-qa1/saved-model/2'
    
    setupPredictor(saved_model_path, modelType='detr', model_weigth_file=os.path.join(saved_model_path, "model.detr"))
    im = cv2.imread(os.path.join(working_directory, 'images', 'IMG_0795.jpg'))  # BGR
    # im = cv2.imread('/workspace-test-v1/mlapp/working_directory_balloon/data/custom/val2017/410488422_5f8991f26e_b.jpg')
    print("im shape", im.shape)
    ret = inference(im, draw=True)
    print("ret", ret)
        
