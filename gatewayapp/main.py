
# Copyright (C) 2020 - 2022 APC, Inc.

import cv2
import os
import paho.mqtt.client as mqtt
import timerworker 
import time
from imageio import imread
import numpy as np
import base64
import io
from PIL import Image
import PIL.ExifTags as ExifTags
import re
import tensorflow as tf

import json
import subprocess
import utility
from threading import Thread
import gpumemory
from shutil import copyfile
from threading import Thread
from glob import glob
import plotmetrics
import shutil
import datetime

import appconfig
import torch

import traceback

from utility import get_base_dir
from utility import get_prefix_topic

WORKING_DIRECTORY = get_base_dir() + "mlapp/working_directory"
def get_working_folder(data_identifier):
    from pathlib import Path
    working_folder = WORKING_DIRECTORY + "/" + data_identifier
    Path(working_folder).mkdir(parents=True, exist_ok=True)    
    return working_folder

# check if we need to do training
g_training_parameter_dict = None
g_model_type = ''
training_param_json_file = os.path.join(WORKING_DIRECTORY, "training_parameter.json")
if os.path.isfile(training_param_json_file):
    with open(training_param_json_file, 'r') as f:
        g_training_parameter_dict = json.load(f)
        print("g_training_parameter_dict", g_training_parameter_dict)
        g_model_type = g_training_parameter_dict["modelType"]

if g_training_parameter_dict is None or (g_training_parameter_dict is not None and g_model_type == 'resnetv2' ):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        tf_memory_limit = 1024*2
        # if appconfig.enable_pretrained_model:
        if g_training_parameter_dict is None:
            tf_memory_limit = 1024
        
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=tf_memory_limit)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

if g_training_parameter_dict is None or (g_training_parameter_dict is not None and g_model_type != 'resnetv2' ):
    # limit only 13GB
    torch.cuda.set_per_process_memory_fraction(8.0/20.0, 0)  #3GB (0.075 * 40)

if appconfig.enable_pretrained_model and g_training_parameter_dict is None:
    import pretraineddetectroninference
    import detrinference as detr_pretrained_inference

if appconfig.enable_custom_model and g_training_parameter_dict is None:
    import customInference

if appconfig.enable_custom_model:
    import bgremoval
    from detectron2.data.datasets import register_coco_instances

if g_training_parameter_dict is not None:
    if g_model_type == 'detr':
        import detrtrain
    elif g_model_type == 'trident':
        import tridenttrain    
    elif g_model_type == 'faster_rcnn':
        import fasterrcnntrain
    elif g_model_type == 'mask_rcnn':
        import maskrcnntrain
    elif g_model_type == 'pointrend':
        import pointrendtrain
    elif g_model_type == 'resnetv2':
        import resnetv2train

g_client = None
g_model = None
g_resnetv2_pretrained_model = None
g_clasess = None
g_training_status = 'idle'
g_training_progress = 0 #0 ~ 100
g_remaining_time_sec = ''
g_max_iterations = 0
g_training_finish_date = datetime.datetime.now()
# g_yolov5_inference_pretrained = yolov5inference.Inference()
g_yolov5_inference_custom = None
g_resnetv2_img_size = 256
g_training_status_time = datetime.datetime.now()


# TODO centralize setting
RESNETV2_TOTAL_EPOCH = 15
SAVED_BASE_PATH = get_base_dir() + "saved-model/"

IMG_HEIGHT = 256
IMG_WIDTH = 256

MQTT_BROKER_HOST = "localhost"
TOPIC_IMAGE_SERVER_HEARTBEAT = get_prefix_topic() + "imageserver/heartbeat"
TOPIC_DELETE_IMAGE = get_prefix_topic() + "deleteimage/#"
TOPIC_PUSH_IMAGE = get_prefix_topic() + "pushimage/#"
TOPIC_COMMAND_START_TRAIN = get_prefix_topic() + "starttrain/#"
TOPIC_COMMAND_TRAIN_STATUS_REQUEST = get_prefix_topic() + "trainstatusrequest"
TOPIC_COMMAND_TRAIN_STATUS = get_prefix_topic() + "trainstatus"
TOPIC_COMMAND_TRAIN_METRIC = get_prefix_topic() + "trainmetric"
TOPIC_COMMAND_TRAIN_REPORT = get_prefix_topic() + "trainreport"
TOPIC_COMMAND_ACTIVATE_MODEL = get_prefix_topic() + "activatemodel"
TOPIC_COMMAND_REMOVE_MODEL = get_prefix_topic() + "removemodel"

TOPIC_COMMAND_DETECT_REQUEST = get_prefix_topic() + "testing/detect/#"
TOPIC_COMMAND_DETECT_RESULT = get_prefix_topic() + "testing/result"

TOPIC_COMMAND_PRETRAINED_DETECT_REQUEST = get_prefix_topic() + "testingpretrained/detect/#"
TOPIC_COMMAND_PRETRAINED_DETECT_RESULT = get_prefix_topic() + "testingpretrained/result"

TOPIC_COMMAND_CANCEL_TRAIN_REQUEST = get_prefix_topic() + "canceltrain/command/#"
TOPIC_COMMAND_CANCEL_TRAIN_RESULT = get_prefix_topic() + "canceltrain/result"

TOPIC_COMMAND_RESTART_ENGINE = get_prefix_topic() + "restartengine"

TOPIC_COMMAND_BG_REMOVAL_REQUEST = get_prefix_topic() + "backgroundremoval/request/#"
TOPIC_COMMAND_BG_REMOVAL_RESULT = get_prefix_topic() + "backgroundremoval/result"


def publish_demoapp_heartbeat(name):
    if g_client is not None:
        g_client.publish(TOPIC_IMAGE_SERVER_HEARTBEAT,"1") 
        # sendTrainingStatus()


def decodeImage(msg):
    topic_elements = msg.topic.split("/")
    filename = topic_elements[-1]
    print("decodeImage filename", filename)
    working_folder = get_working_folder('images')
    file_path =  "{}/{}".format(working_folder, filename)
    with open(file_path, 'wb') as fd:
        fd.write(msg.payload)
        fd.flush()

def deleteImage(msg):
    topic_elements = msg.topic.split("/")
    filename = topic_elements[-1]
    working_folder = get_working_folder('images')
    file_path =  "{}/{}".format(working_folder, filename)
    print("deleteImage filename", filename, "file_path", file_path)
    try:
        os.remove(file_path)
        print("deleted", file_path)
    except OSError:
        pass

def inferenceResnetV2(img_original):
    from tensorflow.keras.preprocessing import image
    global g_model, g_clasess, g_resnetv2_img_size
    if (g_model is None or g_clasess is None):
        return -1, 0
    img_size = g_resnetv2_img_size
    img0 = utility.resize(img_original, int(img_size))
    img = remove_transparency(img0)    

    # img_array = tf.keras.utils.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0)
    # predictions = g_model.predict(img_array)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    predictions = g_model.predict(x)    
    score = tf.nn.softmax(predictions[0])
    print("softmax predictions[0]", score)
    class_index = np.argmax(score)
    class_score = 100 * np.max(score)

    return class_index, int(class_score)

def inferencePretrainedResnetV2(img_original):
    from tensorflow.keras.preprocessing import image

    # img0 = img_original.resize((IMG_WIDTH, IMG_HEIGHT))
    img0 = utility.resize(img_original, 224)
    img = remove_transparency(img0)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)

    predictions = g_resnetv2_pretrained_model.predict(x)    
    top_predictions = tf.keras.applications.resnet_v2.decode_predictions(predictions, top=1)[0]
    print('top_predictions:', top_predictions)
    # top_predictions: [('n02690373', 'airliner', 0.6899511), ('n04266014', 'space_shuttle', 0.3095981), ('n02692877', 'airship', 0.00025558774)]
    if (len(top_predictions) < 1):
        return '{"label":"UNKNOWN", "score":0, "modelType":"resnetV2", "numberOfObjects":0}'
    else:
        score = top_predictions[0][2] * 100
        label = top_predictions[0][1]
        class_score = tf.nn.softmax(predictions[0])
        class_index = np.argmax(class_score)
        with open('./imagenet_class_index.json', 'r') as f:
            json_dict = json.load(f)
            label = label + ":" + json_dict[int(class_index)]['ja']
        return '{"label":"' + label + '", "score":' + str(int(score)) + ', "modelType":"resnetv2", "numberOfObjects":1}'

def inferenceDetectron2(img_original, model_type):
    img_output = pretraineddetectroninference.inference(img_original, model_type)
    return img_output

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)

    # 回転
    orientation = 0
    if image._getexif():
        for k, v in image._getexif().items():
            if k in ExifTags.TAGS and ExifTags.TAGS[k] == "Orientation":
                orientation = v
                break

    if orientation == 2:
        new_image = cv2.flip(new_image, 1)
    elif orientation == 3:
        new_image = cv2.rotate(new_image, cv2.ROTATE_180)
    elif orientation == 4:
        new_image = cv2.flip(new_image, 0)
    elif orientation == 5:
        new_image = cv2.rotate(new_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_image = cv2.flip(new_image, 0)
    elif orientation == 6:
        new_image = cv2.rotate(new_image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        new_image = cv2.rotate(new_image, cv2.ROTATE_90_CLOCKWISE)
        new_image = cv2.flip(new_image, 0)
    elif orientation == 8:
        new_image = cv2.rotate(new_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return new_image

def doDetection(msg, usePretrained):
    global g_model_type
    topic_elements = msg.topic.split("/")
        
    # try: 
    imgdata = base64.b64decode(msg.payload)
    print("imgdata len:", len(imgdata))
    image = Image.open(io.BytesIO(imgdata))
    # img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # print("img shape", img.shape)
    if (not usePretrained):
        dataId = topic_elements[-1]
        print("doDetection dataId", dataId, "payload size", len(msg.payload), "usePretrained", usePretrained)
        image = remove_transparency(image)
        result = '{"numberOfObjects":0, "objects":[], "modelType":"' +  g_model_type + '"}'
        if (g_model_type == "trident" or g_model_type == 'detr' or g_model_type == 'faster_rcnn'
            or g_model_type == 'mask_rcnn' or g_model_type == 'pointrend'): 
            image = pil2cv(image)        
            result = customInference.inference(image)
        elif(g_model_type == 'resnetv2'):
            class_index, score = inferenceResnetV2(image)
            print("class_index", class_index, "score", score)
            # validate g_classes vs class_index
            if(class_index < 0):
                result = '{"label":"' + str(class_index) + '", "score":0, "modelType":"resnetv2", "numberOfObjects":0}'
            elif (class_index < len(g_clasess)):
                result = '{"label":"' + str(g_clasess[class_index]) + '", "score":' + str(score) + ', "modelType":"resnetv2", "numberOfObjects":1}'
            else:
                result = '{"label":"0", "score":0, "modelType":"resnetv2", "numberOfObjects":0}'
        else:
            print("not supported", "modelType", g_model_type)
        g_client.publish(TOPIC_COMMAND_DETECT_RESULT  + '/' + dataId, result)
    else:
        dataId = topic_elements[-2]
        modelType = topic_elements[-1]
        print("doDetection dataId", dataId, "payload size", len(msg.payload), "usePretrained", usePretrained, "modelType", modelType)
        image = remove_transparency(image)
        result = '{"numberOfObjects":0, "objects":[]}'
        
        if (modelType == "detr"):
            # using DETR
            result = detr_pretrained_inference.inference(image)
        elif (modelType == "resnetv2"):
            result = inferencePretrainedResnetV2(image)
            print("pretrained resnetV2 result --> ", result)
        else:
            # modelType is trident and fastrcnn for object detection
            # modelType is pointrend and mask_rcnn for instance segmentation        
            image = pil2cv(image)
            result = pretraineddetectroninference.inference(image, modelType)
        # elif (modelType == "yolov5"):
        #     image = pil2cv(image)        
        #     result = g_yolov5_inference_pretrained.inference(image)            

        g_client.publish(TOPIC_COMMAND_PRETRAINED_DETECT_RESULT  + '/' + dataId, result)
        

def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'='* (4 - missing_padding)
    return base64.b64decode(data, altchars)

def remove_transparency(im, bg_colour=(255, 255, 255)):    
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        print("try to remove alpha channel")
        # # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        # alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGB", im.size, bg_colour)
        bg.paste(im, mask=im.split()[3])
        return bg
    else:
        return im    


def monitorTrainingStatus():
    global g_training_status, g_training_status_time
    if g_training_status:
        diff_time = (datetime.datetime.now() - g_training_status_time).total_seconds()
        if diff_time > 2 * 60: # more than 2 minutes
            print("monitorTrainingStatus diff_time", diff_time, " something is wrong")
            training_parameter_dict = None
            training_param_json_file = os.path.join(WORKING_DIRECTORY, "training_parameter.json")
            if os.path.isfile(training_param_json_file):
                with open(training_param_json_file, 'r') as f:
                    training_parameter_dict = json.load(f)

            if training_parameter_dict is not None:
                # something wrong with the training --> publish cancel training
                modelType = training_parameter_dict["modelType"]
                datasetId = training_parameter_dict["datasetId"]
                msgResponse = '{"status":"cancelled", "model_type":"' + modelType + '", "dataset_id":"' + datasetId  + '"}'
                subprocess.run(["/usr/bin/mosquitto_pub", 
                    "-h", MQTT_BROKER_HOST,
                    "-t", TOPIC_COMMAND_CANCEL_TRAIN_RESULT,
                    "-m", msgResponse])
            
            # restart the engine to clear memory
            remove_training_parameter_json()
            restart()



def trainingCallback(progress, eta_sec):
    global g_training_progress, g_remaining_time_sec, g_training_status_time, g_max_iterations
    g_training_progress = progress
    g_remaining_time_sec = eta_sec
    print("g_training_progress", g_training_progress, "remaining_time_sec", g_remaining_time_sec)
    # sendTrainingStatus() #unfortunately publish message is queued
    gpuinfo = json.dumps(gpumemory.get_gpu_memory_usage())
    print("gpuinfo", gpuinfo)
    msg = '{"status":"' + g_training_status + '", "progress":' + str(g_training_progress) + ', "remain_time":' + str(g_remaining_time_sec) + ', "max_iterations":' + str(g_max_iterations) + ', "gpuinfo":' + gpuinfo + '}'
    subprocess.run(["/usr/bin/mosquitto_pub", 
        "-h", MQTT_BROKER_HOST,
        "-t", TOPIC_COMMAND_TRAIN_STATUS,
        "-m", msg])
    
    g_training_status_time = datetime.datetime.now()

def trainingCallback_resnetV2(epoch, epoch_time_sec):
    global g_training_progress, g_remaining_time_sec    
    g_training_progress = int((epoch+1) * 100.0 / RESNETV2_TOTAL_EPOCH)
    g_remaining_time_sec = epoch_time_sec * (RESNETV2_TOTAL_EPOCH - epoch - 1) + 10
    print("epoch", epoch+1, "g_training_progress", g_training_progress, 
        "epoch_time_sec", epoch_time_sec, "remaining_time_sec", g_remaining_time_sec)
    # sendTrainingStatus() #unfortunately publish message is queued
    gpuinfo = json.dumps(gpumemory.get_gpu_memory_usage())
    print("gpuinfo", gpuinfo)
    msg = '{"status":"' + g_training_status + '", "progress":' + str(g_training_progress) + ', "remain_time":' + str(g_remaining_time_sec) + ', "gpuinfo":' + gpuinfo + '}'
    subprocess.run(["/usr/bin/mosquitto_pub", 
        "-h", MQTT_BROKER_HOST,
        "-t", TOPIC_COMMAND_TRAIN_STATUS,
        "-m", msg])        

def doTraining(msg):
    global g_training_status, g_model

    if g_training_status == 'running':
        print('ignore start training command because g_training_status is ', g_training_status)
        sendTrainingStatus()
        return

    topic_elements = msg.topic.split("/")    

    modelType = topic_elements[-1]
    batchSize = topic_elements[-2]
    imageSize = topic_elements[-3]
    max_iterations = topic_elements[-4]
    datasetId = topic_elements[-5]
    batchSize = "2" # increasing batch may result in  OOM
    print("doTraining imageSize", imageSize, "batchSize", batchSize, "modelType", modelType, "max_iterations", max_iterations, "datasetId", datasetId)
    msgStr = str(msg.payload.decode("utf-8"))
    jsonData = json.loads(msgStr)    
    print(msgStr)
    
    
    if (not jsonData["train"] or not jsonData["val"] or not jsonData["labels"]):
        print("can not start training because data is not enough")
        sendCancel(modelType, datasetId)
        return
    
    # save train.json
    print("write train.json")
    trainJsonFile = open(WORKING_DIRECTORY + "/train.json", "w")
    trainJsonFile.write(json.dumps(jsonData["train"], indent=4))
    trainJsonFile.close()

    # save val.json
    print("write val.json")
    valJsonFile = open(WORKING_DIRECTORY + "/val.json", "w")
    valJsonFile.write(json.dumps(jsonData["val"], indent=4))
    valJsonFile.close()

    # save labels.json
    print("write labels.json")
    # print(json.dumps(jsonData["labels"], indent=4))
    labelsJsonFile = open(WORKING_DIRECTORY + "/labels.json", "w")
    labelsJsonFile.write(json.dumps(jsonData["labels"], indent=4, ensure_ascii=False))
    labelsJsonFile.close()

    # check total images    
    label_ids = [x["id"] for x in jsonData["labels"]]
    total_image_num = 0    
    dir_path = os.path.join(WORKING_DIRECTORY, 'images')
    if(len(label_ids) < 2):
        print("can not start training labels is not enough count_label", len(label_ids))
        sendCancel(modelType, datasetId)
        return

    for label_id in label_ids:
        if modelType == "resnetv2":
            count_train = len([x for x in jsonData["train"]["images"] if x["class_id"] == str(label_id)])
            count_val = len([x for x in jsonData["val"]["images"] if x["class_id"] == str(label_id)])
        else:
            count_train = len([x for x in jsonData["train"]["annotations"] if x["category_id"] == label_id])
            count_val = len([x for x in jsonData["val"]["annotations"] if x["category_id"] == label_id])
        print("modelType", modelType, "count_train", count_train, "count_val", count_val)
        if(count_train  < 1 or count_val < 1):
            print("can not start training because data is not enough count_train", count_train, "count_val", count_val)
            sendCancel(modelType, datasetId)
            return

        if(count_train + count_val < 10):
            print("can not start training because data is not enough count_train+count_val", count_train + count_val)
            sendCancel(modelType, datasetId)
            return
        total_image_num += count_train + count_val

    if(total_image_num < 40):
        print("can not start training because data is not enough total_image_num", total_image_num)
        sendCancel(modelType, datasetId)
        return    
      

    print("start the training")
    img_size = 512
    if (imageSize.isnumeric()):
        img_size = int(imageSize)
    batch_size = 2
    if (batchSize.isnumeric()):
        batch_size = int(batchSize)
    if (max_iterations.isnumeric()):
        max_iterations = int(max_iterations)
    else:
        max_iterations = 500
    
    # save parameter training
    training_parameter = {
        "img_size": img_size,
        "batch_size": batch_size,
        "modelType": modelType,
        "max_iterations": max_iterations,
        "datasetId": datasetId
    }
    training_parameter_json_file = open(os.path.join(WORKING_DIRECTORY, "training_parameter.json"), "w")
    training_parameter_json_file.write(json.dumps(training_parameter, indent=4))
    training_parameter_json_file.close()    

    restart()

def doTrainingThread(img_size, batch_size, modelType, max_iterations, dataset_id):
    global g_training_status, g_training_progress, g_training_finish_date
    print("start doTrainingThread img_size", img_size, "batch_size", batch_size, "modelType", modelType, "max_iterations", max_iterations)
    # train.startTraining(TOTAL_EPOCH, len(g_clasess), trainingCallback)
    # if (modelType == 'trident'):
    if (['trident', 'detr', 'faster_rcnn', 'mask_rcnn', 'pointrend'].count(modelType) > 0):
        try:
            if (modelType == 'detr'):
                detrtrain.main(WORKING_DIRECTORY, img_size, batch_size, max_iterations, trainingCallback,)
            elif (modelType == 'faster_rcnn'):
                fasterrcnntrain.main(WORKING_DIRECTORY, img_size, batch_size, max_iterations, trainingCallback,)
            elif (modelType == 'mask_rcnn'):
                maskrcnntrain.main(WORKING_DIRECTORY, img_size, batch_size, max_iterations, trainingCallback,)
            elif (modelType == 'pointrend'):
                pointrendtrain.main(WORKING_DIRECTORY, img_size, batch_size, max_iterations, trainingCallback,)
            elif (modelType == 'trident'):
                tridenttrain.main(WORKING_DIRECTORY, img_size, batch_size, max_iterations, trainingCallback,)

            # save dataset_id info
            np.savetxt(os.path.join(SAVED_BASE_PATH, 'datasetid.txt'), [dataset_id], fmt='%s')

            saved_base_path = os.path.join(SAVED_BASE_PATH, str(dataset_id))
            if (not os.path.isdir(saved_base_path)):
                os.makedirs(saved_base_path, exist_ok=True)
                print(saved_base_path, "is created")

            # remove any existing model
            old_model_files = glob(saved_base_path  + '/*model*')
            if (len(old_model_files) > 0):
                for old_model_file in old_model_files:
                    os.remove(old_model_file)

            # copy model to saved-model folder
            copyfile(os.path.join(WORKING_DIRECTORY, "output", "model_final.pth"), saved_base_path  + '/model.' + modelType)


            # set new labels
            with open(os.path.join(WORKING_DIRECTORY, "labels.json")) as f:
                labelJson = json.load(f) 
            clasess = np.array([a['id'] for a in labelJson])
            print("clasess", clasess)                            
            np.savetxt(os.path.join(saved_base_path, 'classes.txt'), clasess, fmt='%s')    

            # set new model type
            np.savetxt(os.path.join(saved_base_path, 'modeltype.txt'), [modelType], fmt='%s')   
            
            # draw graph metric
            plotmetrics.plot_metrics(WORKING_DIRECTORY, saved_base_path, labelJson, max_iterations, modelType)
            
        except:
            g_training_status = 'idle'
            g_training_progress = 0
            output_error()
            sendCancel(modelType, dataset_id)
            remove_training_parameter_json()
            restart()
            return

    elif modelType == 'resnetv2':
        saved_base_path = os.path.join(SAVED_BASE_PATH, str(dataset_id))        
        if (not os.path.isdir(saved_base_path)):
            os.makedirs(saved_base_path, exist_ok=True)
            print(saved_base_path, "is created")
        resnetv2train.main(WORKING_DIRECTORY, img_size, batch_size, saved_base_path, RESNETV2_TOTAL_EPOCH, trainingCallback,)

        # save dataset_id info
        np.savetxt(os.path.join(SAVED_BASE_PATH, 'datasetid.txt'), [dataset_id], fmt='%s')

        # set new labels
        with open(os.path.join(WORKING_DIRECTORY, "labels.json")) as f:
            labelJson = json.load(f) 
        clasess = np.array([str(a['id']) for a in labelJson])
        print("clasess", clasess)                        
        np.savetxt(os.path.join(saved_base_path, 'classes.txt'), clasess, fmt='%s')   

        # set new model type
        np.savetxt(os.path.join(saved_base_path, 'modeltype.txt'), [modelType], fmt='%s')   

        # save img size for inference
        np.savetxt(os.path.join(saved_base_path, 'size.txt'), [img_size])
    else:
        print('model type is not found!! modelType:', modelType)
        g_training_status = 'idle'
        g_training_progress = 0
        msgResponse = '{"status":"cancelled", "model_type":"' + modelType + '", "dataset_id":"' + dataset_id  + '"}'
        subprocess.run(["/usr/bin/mosquitto_pub", 
            "-h", MQTT_BROKER_HOST,
            "-t", TOPIC_COMMAND_CANCEL_TRAIN_RESULT,
            "-m", msgResponse])
        remove_training_parameter_json()
        return

    # after training process
    cancel = True
    if (modelType == 'trident'):
        cancel = tridenttrain.g_cancel
    elif (modelType == 'detr'):
        cancel = detrtrain.g_cancel
    elif (modelType == 'faster_rcnn'):
        cancel = fasterrcnntrain.g_cancel
    elif (modelType == 'mask_rcnn'):
        cancel = maskrcnntrain.g_cancel
    elif (modelType == 'pointrend'):
        cancel = pointrendtrain.g_cancel
    elif (modelType == 'resnetv2'):
        cancel = resnetv2train.g_cancel

    if (cancel):
        g_training_status = 'idle'
        g_training_progress = 0
        g_remaining_time_sec = 5 * 60 # 5 min 
        print("training is cancelled")
        # publish cancel
        msgResponse = '{"status":"cancelled", "model_type":"' + modelType + '", "dataset_id":"' + dataset_id  + '"}'
        subprocess.run(["/usr/bin/mosquitto_pub", 
            "-h", MQTT_BROKER_HOST,
            "-t", TOPIC_COMMAND_CANCEL_TRAIN_RESULT,
            "-m", msgResponse])
    else:
        g_training_finish_date = datetime.datetime.now()

        print(g_training_finish_date.isoformat() + " : training is completed")
        msgResponse = '{"status":"completed", "model_type":"' + modelType + '", "dataset_id":"' + dataset_id  + '"}'
        # g_client.publish(TOPIC_COMMAND_TRAIN_STATUS, msgResponse)
        subprocess.run(["/usr/bin/mosquitto_pub", 
            "-h", MQTT_BROKER_HOST,
            "-t", TOPIC_COMMAND_TRAIN_STATUS,
            "-m", msgResponse])
        g_training_status = 'idle'  

        with open(os.path.join(saved_base_path, 'report.png'), "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            # g_client.publish(TOPIC_COMMAND_TRAIN_REPORT, encoded_string)
            subprocess.run(["/usr/bin/mosquitto_pub", 
            "-h", MQTT_BROKER_HOST,
            "-t", TOPIC_COMMAND_TRAIN_REPORT,
            "-m", encoded_string])
    
    # restart the engine to clear memory
    remove_training_parameter_json()
    restart()


def doCancelTraining(msg):
    topic_elements = msg.topic.split("/")
    modelType = topic_elements[-1]
    if (modelType != g_model_type):
        return
    if (modelType == "trident"):
        tridenttrain.cancelTraining()
    elif (modelType == 'faster_rcnn'):
        fasterrcnntrain.cancelTraining()
    elif (modelType == 'detr'):
        detrtrain.cancelTraining();
    elif (modelType == 'mask_rcnn'):
        maskrcnntrain.cancelTraining()
    elif (modelType == 'pointrend'):
        pointrendtrain.cancelTraining()
    elif (modelType == 'resnetv2'):
        resnetv2train.cancelTraining()

def activate_model(msg):    
    dataset_id = str(msg.payload.decode("utf-8"))    
    current_dataset_id = ""
    if (os.path.isfile(os.path.join(SAVED_BASE_PATH, 'datasetid.txt'))):
        current_dataset_id = str(np.loadtxt(os.path.join(SAVED_BASE_PATH, 'datasetid.txt'), dtype=str))
    print("try to activate_model dataset_id:", dataset_id, "current_dataset_id:", current_dataset_id)
    if current_dataset_id != dataset_id:
        saved_base_path = os.path.join(SAVED_BASE_PATH, dataset_id)
        if (os.path.isdir(saved_base_path)):
            # save dataset_id info
            np.savetxt(os.path.join(SAVED_BASE_PATH, 'datasetid.txt'), [dataset_id], fmt='%s')
            # restart the engine to clear memory
            restart()
        else:
            print("datasetId does not exist!")
    else:
        print("ignore activate model command")
    
    

def remove_model(msg):
    dataset_id = str(msg.payload.decode("utf-8"))    
    print("try to remove_model dataset_id:", dataset_id)
    if (os.path.isfile(os.path.join(SAVED_BASE_PATH, 'datasetid.txt'))):
        current_dataset_id = str(np.loadtxt(os.path.join(SAVED_BASE_PATH, 'datasetid.txt'), dtype=str))
        if (dataset_id != current_dataset_id): # prevent delete active model
            saved_base_path = os.path.join(SAVED_BASE_PATH, dataset_id)
            if (os.path.isdir(saved_base_path)):
                try:
                    shutil.rmtree(saved_base_path)
                    print("removed", saved_base_path)
                except OSError as e:
                    print("remove_model error file:%s errno:%s strerror:%s" % (saved_base_path,  e.errno, e.strerror))

def sendTrainingStatus():
    print("training status", g_training_status)
    gpuinfo = json.dumps(gpumemory.get_gpu_memory_usage())
    print("gpuinfo", gpuinfo)
    msg = ''
    if (g_training_status == 'idle'):
        if (os.path.isfile(os.path.join(SAVED_BASE_PATH, 'datasetid.txt'))):
            dataset_id = str(np.loadtxt(os.path.join(SAVED_BASE_PATH, 'datasetid.txt'), dtype=str))
            msg = '{"status":"' + g_training_status + '", "gpuinfo":' + gpuinfo + ',"model_type":"' + g_model_type + '", "dataset_id":"' + dataset_id + '", "finish_date":"' + str(g_training_finish_date) + '"}'
        else:
            msg = '{"status":"' + g_training_status + '", "gpuinfo":' + gpuinfo + ',"model_type":"' + g_model_type + '", "finish_date":"' + str(g_training_finish_date) + '"}'
    elif (g_training_status == 'running'):
        if g_remaining_time_sec != '':
            msg = '{"status":"' + g_training_status + '", "progress":' + str(g_training_progress) + ', "remain_time":' + str(g_remaining_time_sec) + ', "max_iterations":' + str(g_max_iterations) + ', "gpuinfo":' + gpuinfo + '}'
        else:
            msg = '{"status":"' + g_training_status + '", "progress":' + str(g_training_progress) + ', "max_iterations":' + str(g_max_iterations) +  ', "gpuinfo":' + gpuinfo + '}'
    g_client.publish(TOPIC_COMMAND_TRAIN_STATUS, msg) 

def sendCancel(model_type, dataset_id):
    print("training status", "untrained")
    gpuinfo = json.dumps(gpumemory.get_gpu_memory_usage())
    print("gpuinfo", gpuinfo)
    msg = '{"status":"cancelled", "model_type":"' + model_type + '", "dataset_id":"' + dataset_id + '", "gpuinfo":' + gpuinfo + '}'
    g_client.publish(TOPIC_COMMAND_CANCEL_TRAIN_RESULT, msg)

def restart():
    import sys
    print("argv was",sys.argv)
    print("sys.executable was", sys.executable)
    print("restart now")
    
    # import os
    # os.execv(sys.executable, ['python'] +  sys.argv )
    # os.execv(sys.executable, ['engine-' + get_prefix_topic(), 'runengine.sh'])
    import subprocess
    subprocess.Popen( get_base_dir() + 'mlapp/gatewayapp/runengine.sh')
    g_client.disconnect()
    # g_client.loop_stop(force=True)
    # os._exit(0)
    # # quit()
    # print('exit --> you should not see me')

def do_background_removal(msg):
    topic_elements = msg.topic.split("/")
    client_id = topic_elements[-1]
    file_name = topic_elements[-2]
    xywh_str = str(msg.payload.decode("utf-8"))
    print("client_id", client_id, "file_name", file_name, "xywh_str", xywh_str)    
    file_path = os.path.join(WORKING_DIRECTORY, 'images', file_name)
    print("file_path", file_path)
    pil_image = Image.open(file_path)
    points_str = ''
    if (pil_image != None):
        # get polygon of foreground object
        points_str = bgremoval.get_points_str(pil_image, xywh_str)
    
    g_client.publish(TOPIC_COMMAND_BG_REMOVAL_RESULT + '/' + client_id, points_str)

def remove_training_parameter_json():
    json_file = os.path.join(WORKING_DIRECTORY, "training_parameter.json")
    if os.path.isfile(json_file):
        print("remove json file", json_file)
        try:
            # remove the file
            os.remove(json_file)
        except OSError:
            pass


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    training_param_json_file = os.path.join(WORKING_DIRECTORY, "training_parameter.json")
    if os.path.isfile(training_param_json_file):
        sendTrainingStatus()
    
    client.subscribe(TOPIC_PUSH_IMAGE)
    print("subscribed to " + TOPIC_PUSH_IMAGE)
    client.subscribe(TOPIC_DELETE_IMAGE)
    print("subscribed to " + TOPIC_DELETE_IMAGE)
    client.subscribe(TOPIC_COMMAND_START_TRAIN)
    print("subscribed to " + TOPIC_COMMAND_START_TRAIN)
    client.subscribe(TOPIC_COMMAND_TRAIN_STATUS_REQUEST)
    print("subscribed to " + TOPIC_COMMAND_TRAIN_STATUS_REQUEST)
    client.subscribe(TOPIC_COMMAND_DETECT_REQUEST)
    print("subscribed to " + TOPIC_COMMAND_DETECT_REQUEST)
    client.subscribe(TOPIC_COMMAND_PRETRAINED_DETECT_REQUEST)
    print("subscribed to " + TOPIC_COMMAND_PRETRAINED_DETECT_REQUEST)
    client.subscribe(TOPIC_COMMAND_CANCEL_TRAIN_REQUEST)
    print("subscribed to " + TOPIC_COMMAND_CANCEL_TRAIN_REQUEST)
    client.subscribe(TOPIC_COMMAND_RESTART_ENGINE)
    print("subscribed to " + TOPIC_COMMAND_RESTART_ENGINE)
    client.subscribe(TOPIC_COMMAND_BG_REMOVAL_REQUEST)
    print("subscribed to " + TOPIC_COMMAND_BG_REMOVAL_REQUEST)
    client.subscribe(TOPIC_COMMAND_ACTIVATE_MODEL)
    print("subscribed to " + TOPIC_COMMAND_ACTIVATE_MODEL)
    client.subscribe(TOPIC_COMMAND_REMOVE_MODEL)
    print("subscribed to " + TOPIC_COMMAND_REMOVE_MODEL)

def output_error():
    error_time = datetime.datetime.now()
    print("ERROR : " + error_time.isoformat())
    print(traceback.format_exc())


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):    
    global g_model_type
    print(time.strftime("%Y-%m-%d %H:%M:%S.%s"), "topic", msg.topic, " len", len(msg.payload))    
    start_time = time.time()
    if get_prefix_topic() + "pushimage/" in msg.topic:    
        try :
            decodeImage(msg)
        except:
            output_error()

    elif get_prefix_topic() + "deleteimage/" in msg.topic:
        try :
            deleteImage(msg)
        except:
            output_error()

    elif get_prefix_topic() + "starttrain/" in msg.topic:
        try :
            doTraining(msg)
        except:
            output_error()
            topic_elements = msg.topic.split("/")
            modelType = topic_elements[-1]
            datasetId = topic_elements[-5]
            sendCancel(modelType, datasetId)

    elif get_prefix_topic() + "trainstatusrequest" in msg.topic:
        try:
            sendTrainingStatus()
        except:
            output_error()

    elif get_prefix_topic() + "testing/detect" in msg.topic:
        try:
            doDetection(msg, False)
        except:
            output_error()
            topic_elements = msg.topic.split("/")
            dataId = topic_elements[-1]

            result = {}
            result["numberOfObjects"] = 0
            result["objects"] = []
            result["modelType"] = g_model_type

            resultStr = json.dumps(result)

            g_client.publish(TOPIC_COMMAND_DETECT_RESULT  + '/' + dataId, resultStr)

    elif get_prefix_topic() + "testingpretrained/detect" in msg.topic:
        try:
            doDetection(msg, True)
        except:
            output_error()
            topic_elements = msg.topic.split("/")
            dataId = topic_elements[-2]
            modelType = topic_elements[-1]

            result = {}
            result["numberOfObjects"] = 0
            result["objects"] = []
            result["modelType"] = modelType

            resultStr = json.dumps(result)

            g_client.publish(TOPIC_COMMAND_PRETRAINED_DETECT_RESULT  + '/' + dataId, resultStr)

    elif get_prefix_topic() + "canceltrain/command" in msg.topic:
        try:
            doCancelTraining(msg)
        except:
            output_error()

    elif get_prefix_topic() + "restartengine" in msg.topic:
        try:
            restart()
        except:
            output_error()

    elif get_prefix_topic() + "backgroundremoval/request/" in msg.topic:
        try:
            do_background_removal(msg)
        except:
            output_error()
            topic_elements = msg.topic.split("/")
            client_id = topic_elements[-1]
            points_str = ''
            g_client.publish(TOPIC_COMMAND_BG_REMOVAL_RESULT + '/' + client_id, points_str)

    elif get_prefix_topic() + "activatemodel" in msg.topic:
        try:
            activate_model(msg)
        except:
            output_error()

    elif get_prefix_topic() + "removemodel" in msg.topic:
        try:
            remove_model(msg)
        except:
            output_error()
    
    decode_time = time.time() - start_time    
    print(time.strftime("%Y-%m-%d %H:%M:%S.%s"), "topic ", msg.topic, "decode time (seconds)", decode_time)  

if __name__ == '__main__':
    
    g_client = mqtt.Client()
    g_client.on_connect = on_connect
    g_client.on_message = on_message

    g_client.connect(MQTT_BROKER_HOST, 1883, 60)
    
    if (appconfig.enable_pretrained_model and g_training_parameter_dict is None):
        print("try to load pre-trained model")
        pretraineddetectroninference.setupPredictor('mask_rcnn')
        pretraineddetectroninference.setupPredictor('pointrend')

        pretraineddetectroninference.setupPredictor('trident')
        pretraineddetectroninference.setupPredictor('faster_rcnn')
        print("setup predictor ResnetV2")
        g_resnetv2_pretrained_model = tf.keras.applications.ResNet152V2(weights='imagenet')

        detr_pretrained_inference.setupPredictor()            

    if (appconfig.enable_custom_model):
        # bg removal function
        bgremoval.setup()        
        if g_training_parameter_dict is not None:
            print("do the training", g_training_parameter_dict)
            g_training_status = 'running'    
            g_training_status_time = datetime.datetime.now()
            gpuinfo = json.dumps(gpumemory.get_gpu_memory_usage())
            print("gpuinfo", gpuinfo, "g_model_type", g_model_type)
            
            register_coco_instances("my_dataset_train", {}, WORKING_DIRECTORY + "/train.json", WORKING_DIRECTORY + "/images")
            register_coco_instances("my_dataset_val", {}, WORKING_DIRECTORY + "/val.json", WORKING_DIRECTORY + "/images")
            img_size = g_training_parameter_dict["img_size"]
            batch_size = g_training_parameter_dict["batch_size"]
            max_iterations = g_training_parameter_dict["max_iterations"]
            datasetId = g_training_parameter_dict["datasetId"]
            g_max_iterations = max_iterations
            t1 = Thread(target=doTrainingThread, args=(img_size, batch_size, g_model_type, max_iterations, datasetId, ))
            t1.start()
        else:
            print("setup custom model")
            file_dataset_id = os.path.join(SAVED_BASE_PATH, 'datasetid.txt')
            if (os.path.isfile(file_dataset_id)):
                dataset_id = str(np.loadtxt(os.path.join(SAVED_BASE_PATH, 'datasetid.txt'), dtype=str))
                saved_base_path = os.path.join(SAVED_BASE_PATH, dataset_id)
                print("saved_base_path model is ", saved_base_path)
                model_type_path = os.path.join(saved_base_path, 'modeltype.txt')
                g_model_type = str(np.loadtxt(model_type_path, dtype=str))
                print("custom model is ", g_model_type, model_type_path)

                if (g_model_type == 'trident' or g_model_type == 'detr' or g_model_type == 'faster_rcnn'
                        or g_model_type == 'mask_rcnn' or g_model_type == 'pointrend'):
                    customInference.setupPredictor(saved_base_path, g_model_type)
                elif (g_model_type == 'resnetv2'):
                    resnetv2_model_path = os.path.join(saved_base_path, 'my_model.h5')
                    g_model = tf.keras.models.load_model(resnetv2_model_path)
                    classes_path = os.path.join(saved_base_path, 'classes.txt')
                    g_clasess = np.loadtxt(classes_path, dtype=str)
                    print("loaded model resnetv2", resnetv2_model_path)
                    resnetv2_img_size_path = os.path.join(saved_base_path, 'size.txt')
                    g_resnetv2_img_size = int(np.loadtxt(resnetv2_img_size_path) + 0)
                    print("resnetv2_img_size_path", resnetv2_img_size_path, "size", g_resnetv2_img_size) 
        

    # engine heartbeat every 20 seconds
    engine_heartbeat_rt = timerworker.RepeatedTimer(20, publish_demoapp_heartbeat, "demoapp-heartbeat") # it auto-starts, no need of rt.start()

    if g_training_parameter_dict is not None:
        # start monitoring training
        monitoring_training_status_rt = timerworker.RepeatedTimer(20, monitorTrainingStatus) # it auto-starts, no need of rt.start()

    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    g_client.loop_forever()

    # stop engine heartbeat
    engine_heartbeat_rt.stop()

    if g_training_parameter_dict is not None:
        # stop monitoring training status
        monitoring_training_status_rt.stop()