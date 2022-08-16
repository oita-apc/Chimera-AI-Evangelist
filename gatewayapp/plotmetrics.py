# Copyright (C) 2020 - 2022 APC, Inc.

import json
import matplotlib.pyplot as plt
import os
import numpy as np
import japanize_matplotlib 



def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def plot_metrics(working_directory, saved_model_path, label_json, max_iterations, model_type):
    number_of_columns = len(label_json) + 1
    print("number_of_columns", number_of_columns, "max_iterations", max_iterations, "model_type", model_type)
    experiment_metrics = load_json_arr(os.path.join(working_directory, "output", "metrics.json"))
    
    x_axis = np.arange(0, max_iterations + 1, max_iterations/5)
    x_axis = x_axis.astype(int)

    key_metric = ''
    if (model_type == 'detr' or model_type == 'faster_rcnn' or model_type == 'trident'):
        key_metric = 'bbox/AP'
    elif (model_type == 'mask_rcnn' or model_type == 'pointrend'):
        key_metric = 'segm/AP'

    if (key_metric == ''):
        print('model_type', model_type, ' is not supported')
        return

    plt.figure(figsize=(10, 7))    
    column_no = 1
    plt.subplot(1,number_of_columns, column_no)
    plt.xticks(x_axis)
    ax = plt.gca()
    ax.set_ylim([0, 100])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if key_metric in x.keys()], 
        [x[key_metric] for x  in experiment_metrics if key_metric in x.keys()])
    if (number_of_columns > 2):
        plt.title(key_metric + ' Overall')    
    else:
        plt.title(key_metric + ' ' + label_json[0]['name'])    
    plt.xlabel('iteration')

    if (number_of_columns > 2):
        for label in label_json:
            column_no = column_no + 1
            label_id = label['id']
            plt.subplot(1, number_of_columns, column_no)
            plt.xticks(x_axis)
            ax = plt.gca()
            ax.set_ylim([0, 100])
            plt.plot(
                [x['iteration'] for x in experiment_metrics if key_metric in x.keys()], 
                [x[key_metric + '-' + str(label_id)] for x  in experiment_metrics if key_metric in x.keys()])
            plt.title(key_metric + ' ' + label['name'])
            plt.xlabel('iteration')
    plt.savefig(os.path.join(saved_model_path, 'report.png'))

if __name__ == '__main__':
    working_directory = "/workspace-test-v1/mlapp/working_directory"
    saved_model_path = "/workspace-test-v1/saved-model"
    max_iteration = 600
    model_type = 'mask_rcnn'
    with open(os.path.join(working_directory, "labels.json")) as f:
        labelJson = json.load(f)
    print("labelJson", labelJson)
    plot_metrics(working_directory, saved_model_path, labelJson, max_iteration, model_type)