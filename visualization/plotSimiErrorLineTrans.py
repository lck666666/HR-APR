import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.functions import *

APRS = ['DFNet', "MS-Transformer", "PoseNet"]

def main(parser):
    SCENES_CAM =  ['KingsCollege','OldHospital','ShopFacade','StMarysChurch']
    simi_array = []
    pos_error_array = []
    ori_error_array = []
    labels = []
    args = parser.parse_args()
    apr = args.apr
    results_path = f'../{apr}/results'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        print(f"Folder '{results_path}' has been created.")
    else:
        print(f"Folder '{results_path}' already exists.")
    for scene in SCENES_CAM:
        test_gt_file =f'../{apr}/Cambridge/{scene}_test_gt.txt'
        pred_file = f'../{apr}/Cambridge/{scene}_predict.txt'
        dist_th = 1.5
        dist_str = str(dist_th).replace('.','') + '0'
        test_simi_file = f'../{apr}/simi_ranking/{scene}_simi_{dist_str}.txt'
        test_list = get_test_file(test_gt_file)
        pred_list = []
        with open(pred_file) as f:
            for line in f.readlines():
                data = line.split('/t/n')
                for stri in data:
                    sub_str = stri.split(' ')
                    pred_list.append(sub_str)
        simi_list = []
        with open(test_simi_file, 'r') as f:
            for line in f.readlines():
                data = line.split('/t/n')
                for stri in data:
                    sub_str = stri.split(' ')
                    simi_list.append(sub_str)
        simi_dict = {}
        for i in simi_list:
            simi_dict[i[0]] = float(i[1])
        
        pos_true_dict = {}
        pos_out_dict = {}
        for key in simi_dict.keys():
            pos_true_dict[key] = getPos(key,test_gt_file)
            pos_out_dict[key] = getPredictPos(key,pred_file)

        for key in simi_dict.keys():
            pos_true,ori_true = pos_true_dict[key]
            pos_out, ori_out= pos_out_dict[key]
            pos_error = np.linalg.norm(pos_true - pos_out)
            R_out = ToR(ori_out)
            R_true = ToR(ori_true)
            ori_error = np.rad2deg(np.arccos(((R_out.T@R_true).trace() - 1)*0.5))
            simi_array.append(simi_dict[key])
            pos_error_array.append(pos_error)
            ori_error_array.append(ori_error)
            labels.append(scene)

    simi_threshold_list = np.array([0,0.2,0.4,0.6,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99])
    df = pd.DataFrame(dict(x=pos_error_array, y=simi_array, label=labels))
    groups = df.groupby('label')
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(24,10),constrained_layout=True)
    plt.rcParams.update({'font.size': 35})
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)
    ax1.tick_params(axis='both', which='major', labelsize=35)
    ax2.tick_params(axis='both', which='major', labelsize=35)
    for name,group in groups:
        norm_ori_error = []
        for threshold in simi_threshold_list:
            error_list = []
            y = group.y.tolist()
            x = group.x.tolist()
            for i in range(len(y)):
                if y[i] >= threshold:
                    error_list.append(x[i])
            norm_ori_error.append(np.mean(error_list))
        norm_max = max(norm_ori_error)
        norm_ori_error  = np.array(norm_ori_error)/norm_max
        ax1.plot(simi_threshold_list,norm_ori_error,label = name)
    ax1.legend()
    ax1.set_xlabel("Similarity threshold", fontsize=35)
    ax1.set_ylabel("Normalized mean translation error",fontsize=35)
    #plt.tight_layout()
    ax1.set_ylim(0, 1)

    SCENES_7S = ['chess','fire','heads','office','pumpkin','redkitchen','stairs']
    simi_array = []
    pos_error_array = []
    ori_error_array = []
    labels = []
    for scene in SCENES_7S:
        test_gt_file =f'../{apr}/7Scenes/{scene}_test_gt.txt'
        pred_file = f'../{apr}/7Scenes/{scene}_predict.txt'
        dist_th = 0.2
        dist_str = str(dist_th).replace('0.','') + '0'
        test_simi_file = f'../{apr}/simi_ranking/{scene}_simi_{dist_str}.txt'
        test_list = get_test_file(test_gt_file)
        pred_list = []
        with open(pred_file) as f:
            for line in f.readlines():
                data = line.split('/t/n')
                for stri in data:
                    sub_str = stri.split(' ')
                    pred_list.append(sub_str)
        simi_list = []
        with open(test_simi_file, 'r') as f:
            for line in f.readlines():
                data = line.split('/t/n')
                for stri in data:
                    sub_str = stri.split(' ')
                    simi_list.append(sub_str)
        simi_dict = {}
        for i in simi_list:
            simi_dict[i[0]] = float(i[1])
        
        pos_true_dict = {}
        pos_out_dict = {}
        for key in simi_dict.keys():
            pos_true_dict[key] = getPos(key,test_gt_file)
            pos_out_dict[key] = getPredictPos(key,pred_file)

        for key in simi_dict.keys():
            pos_true,ori_true = pos_true_dict[key]
            pos_out, ori_out= pos_out_dict[key]
            pos_error = np.linalg.norm(pos_true - pos_out)
            R_out = ToR(ori_out)
            R_true = ToR(ori_true)
            ori_error = np.rad2deg(np.arccos(((R_out.T@R_true).trace() - 1)*0.5))
            simi_array.append(simi_dict[key])
            pos_error_array.append(pos_error)
            ori_error_array.append(ori_error)
            labels.append(scene)

    simi_threshold_list = np.array([0,0.2,0.4,0.6,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99])
    df = pd.DataFrame(dict(x=pos_error_array, y=simi_array, label=labels))
    groups = df.groupby('label')
    for name,group in groups:
        norm_ori_error = []
        for threshold in simi_threshold_list:
            error_list = []
            y = group.y.tolist()
            x = group.x.tolist()
            for i in range(len(y)):
                if y[i] >= threshold:
                    error_list.append(x[i])
            norm_ori_error.append(np.mean(error_list))
        norm_max = max(norm_ori_error)
        norm_ori_error  = np.array(norm_ori_error)/norm_max
        ax2.plot(simi_threshold_list,norm_ori_error,label = name)
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Similarity threshold",fontsize=35)
    plt.savefig(f'../{apr}/results/trans_correlation_line.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apr", default=APRS[0], choices=APRS)
    main(parser)