#!/usr/bin/env python3
# Copyright (C) 2020 - 2022 APC, Inc.

"""
mask r-cnn using detectron
"""

import json
import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances


from detectron2.engine.hooks import HookBase
import time
import datetime
from timeit import default_timer as timer
from detectron2.utils.events import EventStorage, get_event_storage
import shutil
from pathlib import Path

EVAL_PERIOD = 20

g_cancel = False
g_prev_end_time = None

class TrainingCallback(HookBase):

    def __init__(self, max_iter, publish_func):
        super(TrainingCallback, self).__init__()
        self._publish_func = publish_func
        self._max_iter = max_iter
    
    def before_step(self):
        global g_prev_end_time
        if (g_prev_end_time is None):
            g_prev_end_time = datetime.datetime.now()

    def after_step(self):
        global g_prev_end_time
        iter_done = self.trainer.storage.iter - self.trainer.start_iter + 1
        if (iter_done % EVAL_PERIOD == 0):
            # storage = get_event_storage()
            # try:
            #     eta_seconds = storage.history("time").median(1000) * (self._max_iter - iter_done - 1)
            #     print("eta_seconds", eta_seconds, " -> ", str(datetime.timedelta(seconds=int(eta_seconds))))
            #     self._publish_func(int(iter_done * 100.0 / self._max_iter), eta_seconds)
            # except KeyError:
            #     print('NO KEY')
            diff_time = (datetime.datetime.now() - g_prev_end_time).total_seconds()
            progress = int((iter_done + 1) * 100.0 / self._max_iter)
            if (progress > 5):
                progress = progress - 1
            eta_sec = int(diff_time * (self._max_iter - (iter_done + 1))/EVAL_PERIOD) + 20
            print(datetime.datetime.now(), "eta_seconds", eta_sec, " -> ", str(datetime.timedelta(seconds=int(eta_sec))))
            self._publish_func(progress, eta_sec)
            g_prev_end_time = datetime.datetime.now()
        if (g_cancel):
            print("try to stop training")
            raise ValueError('Training is cancelled')

class Trainer(DefaultTrainer):
    _current_iter = 0
    def __init__(self, cfg, callback):
        self._callback = callback
        super(Trainer, self).__init__(cfg)        

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):        
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,TrainingCallback(self.max_iter, self._callback))
        return hooks 
    
    def after_step(self):
        super().after_step()
        self.__class__._current_iter = self.iter + 1        

def setup(working_directory, image_size, batch_size, max_iterations):
    args = default_argument_parser().parse_args()
    """
    Create configs and perform basic setups.
    """
    # register_coco_instances("my_dataset_train", {}, "/home/buser/project/workspace-matterport/dataset_production/train698.json", "/home/buser/project/workspace-matterport/dataset_production/dataset/images")
    # register_coco_instances("my_dataset_val", {}, "/home/buser/project/workspace-matterport/dataset_production/val698.json", "/home/buser/project/workspace-matterport/dataset_production/dataset/images")

    # register_coco_instances("my_dataset_train", {}, working_directory + "/train.json", working_directory + "/images")
    # register_coco_instances("my_dataset_val", {}, working_directory + "/val.json", working_directory + "/images")

    cfg = get_cfg()

    cfg.merge_from_file('./maskrcnn/configs/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = './maskrcnn/model/model_final_f10217.pkl' #mask_rcnn_R_50_FPN_3x
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = max_iterations    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    # cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
    
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.COLOR_AUG_SSD = True
    
    # cfg.INPUT.MIN_SIZE_TRAIN = (1024, 1280, 1600, 1920, 2240,)
    cfg.INPUT.MIN_SIZE_TRAIN = (image_size,)
    if (image_size == 256):
        cfg.INPUT.MIN_SIZE_TRAIN = (256, )
    elif (image_size == 512):    
        cfg.INPUT.MIN_SIZE_TRAIN = (256, 384, 512, )
    elif (image_size == 1024):
        cfg.INPUT.MIN_SIZE_TRAIN = (256, 384, 512, 640, 720, 1024, )
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MAX_SIZE_TRAIN = image_size
    cfg.INPUT.MIN_SIZE_TEST = image_size
    cfg.INPUT.MAX_SIZE_TEST = image_size
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)

    with open(os.path.join(working_directory, "labels.json")) as f:
        labelJson = json.load(f) 
    print(labelJson)
    number_classes = len(labelJson)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_classes  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    
    cfg.OUTPUT_DIR = os.path.join(working_directory, "output")
    # shutil.rmtree(cfg.OUTPUT_DIR) # clean up first
    dirpath = Path(cfg.OUTPUT_DIR)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    else:
        dirpath.mkdir(parents=True, exist_ok=True)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(working_directory, image_size, batch_size, max_iterations, dummyCallbackParam):
    global g_cancel
    g_cancel = False
    print("working_directory", working_directory, "image_size", image_size, "batch_size", batch_size, "max_iterations", max_iterations)
    cfg = setup(working_directory, image_size, batch_size, max_iterations)        

    trainer = Trainer(cfg, dummyCallbackParam)
    trainer.resume_or_load(resume=False)
    return trainer.train()

def cancelTraining():
    global g_cancel
    print("try to cancel training trident")
    g_cancel = True

def dummyCallback(progress, eta_sec):
    # print("dummyCallback progress(%)", progress, "eta_sec", eta_sec)   
    pass

if __name__ == "__main__":
    import torch
    torch.cuda.set_per_process_memory_fraction(0.4, 0)

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    image_size = 512 # 256 512 1024
    batch_size = 2 # 8, 16, 32 --> will crash OOM
    max_iterations = 10
    # working_directory = "/home/buser/project/workspace-demoapp-v2/mlapp/working_directory"
    
    try: 
        working_directory = '/workspace-test-v1/mlapp/working_directory_is1'
        register_coco_instances("my_dataset_train", {}, working_directory + "/train.json", working_directory + "/images")
        register_coco_instances("my_dataset_val", {}, working_directory + "/val.json", working_directory + "/images")

        # working_directory = '/workspace-test-v1/mlapp/working_directory_balloon'
        # register_coco_instances("my_dataset_train", {}, working_directory + "/train.json", working_directory + "/data/custom/train2017")
        # register_coco_instances("my_dataset_val", {}, working_directory + "/val.json", working_directory + "/data/custom/val2017")

        main(working_directory, image_size, batch_size, max_iterations, dummyCallback)

        with open(os.path.join(working_directory, "labels.json")) as f:
                    labelJson = json.load(f) 
        import plotmetrics
        plotmetrics.plot_metrics(working_directory, working_directory, labelJson, max_iterations, 'mask_rcnn')
    except ValueError as err:
        print(err.args)
