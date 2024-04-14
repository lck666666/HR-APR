import argparse
import numpy as np
from utils.functions import *


SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
APRS = ['DFNet', "MS-Transformer", "PoseNet"]

def main(parser):
    args = parser.parse_args()
    scene = args.scene
    apr = args.apr
    test_gt_file = f'../{apr}/7Scenes/{scene}_test_gt.txt'
    pred_file = f'../{apr}/7Scenes/{scene}_predict.txt'
    simi_th = float(args.gamma)
    dist_th = 0.2
    dist_str = str(dist_th).replace('0.','') + '0'
    test_simi_file = f'../{apr}/simi_ranking/df_{scene}_simi_{dist_str}.txt'
    nefes50_file = f'../{apr}/7Scenes/{scene}_Nefes_50.txt'
    nefes10_file = f'../{apr}/7Scenes/{scene}_Nefes_10.txt'
    test_list = get_test_file(test_gt_file)

    pred_dict = {}
    nefes50_dict = {}
    nefes10_dict = {}
    simi_dict = {}

    for img in test_list:
        pos_out, ori_out = getPredictPos(img,pred_file)
        pos_refine50,ori_refine50 = getPos(img,nefes50_file)
        pos_refine10,ori_refine10 = getPos(img,nefes10_file)

        pos_out = np.array(pos_out)
        pos_refine50 = np.array(pos_refine50)
        pos_refine10 = np.array(pos_refine10)
        ori_out = ori_out / np.sqrt(np.sum(ori_out**2))
        ori_refine50 = ori_refine50 / np.sqrt(np.sum(ori_refine50**2))
        ori_refine10 = ori_refine10 / np.sqrt(np.sum(ori_refine10**2))

        pred_dict[img] = [pos_out,ori_out]
        nefes50_dict[img] = [pos_refine50,ori_refine50]
        nefes10_dict[img] = [pos_refine10,ori_refine10]
        simi_dict[img] = getSimi(img,test_simi_file)
    
    results = []
    results_retain = []
    retain_num = 0
    for img in test_list:
        if simi_dict[img] >= simi_th:
            retain_num +=1
            pos_out_wo = pred_dict[img][0]
            ori_out_wo = pred_dict[img][1]
            pos_out = nefes10_dict[img][0]
            ori_out = nefes10_dict[img][1]
            pos_true,ori_true = getPos(img,test_gt_file)
            pos_error = np.linalg.norm(pos_true - pos_out)
            pos_error_wo = np.linalg.norm(pos_true - pos_out_wo)
            R_out = ToR(ori_out)
            R_out_wo = ToR(ori_out_wo)
            R_true = ToR(ori_true)
            ori_error = np.rad2deg(np.arccos(((R_out.T@R_true).trace() - 1)*0.5))
            ori_error_wo = np.rad2deg(np.arccos(((R_out_wo.T@R_true).trace() - 1)*0.5))
            results.append([pos_error, ori_error])
            results_retain.append([pos_error_wo, ori_error_wo])
        elif simi_dict[img] < simi_th:
            pos_refine50 = nefes50_dict[img][0]
            ori_refine50 = nefes50_dict[img][1]
            pos_true,ori_true = getPos(img,test_gt_file)
            pos_error = np.linalg.norm(pos_true - pos_refine50)
            R_refine50 = ToR(ori_refine50)
            R_true = ToR(ori_true)
            ori_error = np.rad2deg(np.arccos(((R_refine50.T@R_true).trace() - 1)*0.5))
            results.append([pos_error, ori_error])
    pos_err_arr = []
    ori_err_arr = []
    num_low_precision = 0
    num_mid_precision = 0
    num_high_precision = 0

    num_5cm_5deg = 0
    for i in range(len(results_retain)):
        pos_err_arr.append(results_retain[i][0])
        ori_err_arr.append(results_retain[i][1])
        if results_retain[i][0] <= 0.05 and results_retain[i][1] <= 5:
            num_5cm_5deg += 1

        if results_retain[i][0] <=5 and results_retain[i][1] <= 10:
            num_low_precision += 1
            if results_retain[i][0] <=0.5 and results_retain[i][1] <= 5:
                num_mid_precision += 1
                if results_retain[i][0] <=0.25 and results_retain[i][1] <= 2:
                    num_high_precision += 1
    pos_err_arr = np.array(pos_err_arr)
    ori_err_arr = np.array(ori_err_arr)
    # standard log
    print("---------------------------HR-APR+filter---------------------------")
    print("retained ratio: ",retain_num/len(results))
    #print("5cm 5deg: ",num_5cm_5deg/len(pos_err_arr))

    print('0.25m, 2degree: ',num_high_precision/len(pos_err_arr))
    print('0.5m, 5degree: ',num_mid_precision/len(pos_err_arr))
    print('5m, 10degree: ',num_low_precision/len(pos_err_arr))
    print('Q1 quantile of pos error: ', np.quantile(pos_err_arr,.25))
    print('Q3 quantile of pos error: ', np.quantile(pos_err_arr,.75))
    print('Q1 quantile of ori error: ', np.quantile(ori_err_arr,.25))
    print('Q3 quantile of ori error: ', np.quantile(ori_err_arr,.75))
    print ('Median error {}m and {} degrees.'.format(np.median(pos_err_arr), np.median(ori_err_arr)))
    print ('Mean error {}m and {} degrees.'.format(np.mean(pos_err_arr), np.mean(ori_err_arr)))

    pos_err_arr = []
    ori_err_arr = []
    num_low_precision = 0
    num_mid_precision = 0
    num_high_precision = 0

    num_5cm_5deg = 0
    for i in range(len(results)):
        pos_err_arr.append(results[i][0])
        ori_err_arr.append(results[i][1])
        if results[i][0] <= 0.05 and results[i][1] <= 5:
            num_5cm_5deg += 1

        if results[i][0] <=5 and results[i][1] <= 10:
            num_low_precision += 1
            if results[i][0] <=0.5 and results[i][1] <= 5:
                num_mid_precision += 1
                if results[i][0] <=0.25 and results[i][1] <= 2:
                    num_high_precision += 1
    pos_err_arr = np.array(pos_err_arr)
    ori_err_arr = np.array(ori_err_arr)
    # standard log
    print("---------------------------HR-APR---------------------------")
    print("image number: ",len(pos_err_arr))
    #print("5cm 5deg: ",num_5cm_5deg/len(pos_err_arr))

    print('0.25m, 2degree: ',num_high_precision/len(pos_err_arr))
    print('0.5m, 5degree: ',num_mid_precision/len(pos_err_arr))
    print('5m, 10degree: ',num_low_precision/len(pos_err_arr))
    print('Q1 quantile of pos error: ', np.quantile(pos_err_arr,.25))
    print('Q3 quantile of pos error: ', np.quantile(pos_err_arr,.75))
    print('Q1 quantile of ori error: ', np.quantile(ori_err_arr,.25))
    print('Q3 quantile of ori error: ', np.quantile(ori_err_arr,.75))
    print ('Median error {}m and {} degrees.'.format(np.median(pos_err_arr), np.median(ori_err_arr)))
    print ('Mean error {}m and {} degrees.'.format(np.mean(pos_err_arr), np.mean(ori_err_arr)))

    text = []
    results = []
    with open(pred_file) as f:
        for line in f.readlines():
            data = line.split('/t/n')
            for stri in data:
                sub_str = stri.split(' ')
                text.append(sub_str)
    for i in range(len(text)):
        pos_true,ori_true = getPos(text[i][0],test_gt_file)
        pos_out, ori_out= getPredictPos(text[i][0],pred_file)
        pos_error = np.linalg.norm(pos_true - pos_out)
        R_out = ToR(ori_out)
        R_true = ToR(ori_true)
        error_ori = np.rad2deg(np.arccos(((R_out.T@R_true).trace() - 1)*0.5))
        results.append([pos_error,error_ori])
    pos_err_arr = []
    ori_err_arr = []
    num_low_precision = 0
    num_mid_precision = 0
    num_high_precision = 0
    num_5cm_5deg = 0
    for i in range(len(results)):
        pos_err_arr.append(results[i][0])
        ori_err_arr.append(results[i][1])
        if results[i][0] <=0.05 and results[i][1] <= 5:
            num_5cm_5deg += 1
        if results[i][0] <=5 and results[i][1] <= 10:
            num_low_precision += 1
            if results[i][0] <=0.5 and results[i][1] <= 5:
                num_mid_precision += 1
                if results[i][0] <=0.25 and results[i][1] <= 2:
                    num_high_precision += 1
    pos_err_arr = np.array(pos_err_arr)
    ori_err_arr = np.array(ori_err_arr)
    # standard log
    print("---------------------------Original Results---------------------------")
    print("image number: ",len(pos_err_arr))
    #print("5cm 5deg: ",num_5cm_5deg/len(pos_err_arr))

    print('0.25m, 2degree: ',num_high_precision/len(pos_err_arr))
    print('0.5m, 5degree: ',num_mid_precision/len(pos_err_arr))
    print('5m, 10degree: ',num_low_precision/len(pos_err_arr))
    print('Q1 quantile of pos error: ', np.quantile(pos_err_arr,.25))
    print('Q3 quantile of pos error: ', np.quantile(pos_err_arr,.75))
    print('Q1 quantile of ori error: ', np.quantile(ori_err_arr,.25))
    print('Q3 quantile of ori error: ', np.quantile(ori_err_arr,.75))
    print ('Median error {}m and {} degrees.'.format(np.median(pos_err_arr), np.median(ori_err_arr)))
    print ('Mean error {}m and {} degrees.'.format(np.mean(pos_err_arr), np.mean(ori_err_arr)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apr", default=APRS[0], choices=APRS)
    parser.add_argument("--scene", default=SCENES[0], choices=SCENES)
    parser.add_argument("--gamma", default=0.95)
    main(parser)
