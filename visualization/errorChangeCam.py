import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils.functions import *


APRS = ['DFNet', "MS-Transformer", "PoseNet"]

def main(parser):
    args = parser.parse_args()
    apr = args.apr
    simi_th = args.gamma
    SCENES = ["KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
    results_path = f'../{apr}/results'
    if not os.path.exists(results_path):
    # 如果不存在，则创建文件夹
        os.makedirs(results_path)
        print(f"Folder '{results_path}' has been created.")
    else:
        print(f"Folder '{results_path}' already exists.")
    for scene in SCENES:
        test_gt_file = f'../{apr}/Cambridge/{scene}_test_gt.txt'
        pred_file = f'../{apr}/Cambridge/{scene}_predict.txt'
        dist_th = 1.5
        dist_str = str(dist_th).replace('.','') + '0'
        test_simi_file = f'../{apr}/simi_ranking/{scene}_simi_{dist_str}.txt'
        nefes50_file = f'../{apr}/Cambridge/{scene}_Nefes_50.txt'
        nefes30_file = f'../{apr}/Cambridge/{scene}_Nefes_30.txt'
        nefes10_file = f'../{apr}/Cambridge/{scene}_Nefes_10.txt'
        test_list = get_test_file(test_gt_file)

        pred_dict = {}
        nefes50_dict = {}
        nefes30_dict = {}
        nefes10_dict = {}
        simi_dict = {}

        for img in test_list:
            pos_out, ori_out = getPredictPos(img,pred_file)
            pos_refine50,ori_refine50 = getPos(img,nefes50_file)
            pos_refine30,ori_refine30 = getPos(img,nefes30_file)
            pos_refine10,ori_refine10 = getPos(img,nefes10_file)

            pos_out = np.array(pos_out)
            pos_refine50 = np.array(pos_refine50)
            pos_refine30 = np.array(pos_refine30)
            pos_refine10 = np.array(pos_refine10)
            ori_out = ori_out / np.sqrt(np.sum(ori_out**2))
            ori_refine50 = ori_refine50 / np.sqrt(np.sum(ori_refine50**2))
            ori_refine30 = ori_refine30 / np.sqrt(np.sum(ori_refine30**2))
            ori_refine10 = ori_refine10 / np.sqrt(np.sum(ori_refine10**2))

            pred_dict[img] = [pos_out,ori_out]
            nefes50_dict[img] = [pos_refine50,ori_refine50]
            nefes30_dict[img] = [pos_refine30,ori_refine30]
            nefes10_dict[img] = [pos_refine10,ori_refine10]
            simi_dict[img] = getSimi(img,test_simi_file)
        
        hs_results = []
        hs_10_results = []
        hs_30_results = []
        hs_50_results = []

        ls_results = []
        ls_10_results = []
        ls_30_results = []
        ls_50_results = []
        for img in test_list:
            if simi_dict[img] >= simi_th:
                pos_out = pred_dict[img][0]
                ori_out = pred_dict[img][1]
                pos_refine10 = nefes10_dict[img][0]
                ori_refine10 = nefes10_dict[img][1]
                pos_refine30 = nefes30_dict[img][0]
                ori_refine30 = nefes30_dict[img][1]
                pos_refine50 = nefes50_dict[img][0]
                ori_refine50 = nefes50_dict[img][1]
                pos_true,ori_true = getPos(img,test_gt_file)

                pos_error_hs = np.linalg.norm(pos_true - pos_out)
                R_out = ToR(ori_out)
                R_true = ToR(ori_true)
                ori_error_hs = np.rad2deg(np.arccos(((R_out.T@R_true).trace() - 1)*0.5))
                hs_results.append([pos_error_hs, ori_error_hs])

                pos_error_hs10 = np.linalg.norm(pos_true - pos_refine10)
                R_refine10 = ToR(ori_refine10)
                R_true = ToR(ori_true)
                ori_error_hs10 = np.rad2deg(np.arccos(((R_refine10.T@R_true).trace() - 1)*0.5))
                hs_10_results.append([pos_error_hs10, ori_error_hs10])

                pos_error_hs30 = np.linalg.norm(pos_true - pos_refine30)
                R_refine30 = ToR(ori_refine30)
                R_true = ToR(ori_true)
                ori_error_hs30 = np.rad2deg(np.arccos(((R_refine30.T@R_true).trace() - 1)*0.5))
                hs_30_results.append([pos_error_hs30, ori_error_hs30])

                pos_error_hs50 = np.linalg.norm(pos_true - pos_refine50)
                R_refine50 = ToR(ori_refine50)
                R_true = ToR(ori_true)
                ori_error_hs50 = np.rad2deg(np.arccos(((R_refine50.T@R_true).trace() - 1)*0.5))
                hs_50_results.append([pos_error_hs50, ori_error_hs50])

            elif simi_dict[img] < simi_th:
                pos_out = pred_dict[img][0]
                ori_out = pred_dict[img][1]
                pos_refine10 = nefes10_dict[img][0]
                ori_refine10 = nefes10_dict[img][1]
                pos_refine30 = nefes30_dict[img][0]
                ori_refine30 = nefes30_dict[img][1]
                pos_refine50 = nefes50_dict[img][0]
                ori_refine50 = nefes50_dict[img][1]
                pos_true,ori_true = getPos(img,test_gt_file)

                pos_error_ls = np.linalg.norm(pos_true - pos_out)
                R_out = ToR(ori_out)
                R_true = ToR(ori_true)
                ori_error_ls = np.rad2deg(np.arccos(((R_out.T@R_true).trace() - 1)*0.5))
                ls_results.append([pos_error_ls, ori_error_ls])

                pos_error_ls10 = np.linalg.norm(pos_true - pos_refine10)
                R_refine10 = ToR(ori_refine10)
                R_true = ToR(ori_true)
                ori_error_ls10 = np.rad2deg(np.arccos(((R_refine10.T@R_true).trace() - 1)*0.5))
                ls_10_results.append([pos_error_ls10, ori_error_ls10])

                pos_error_ls30 = np.linalg.norm(pos_true - pos_refine30)
                R_refine30 = ToR(ori_refine30)
                R_true = ToR(ori_true)
                ori_error_ls30 = np.rad2deg(np.arccos(((R_refine30.T@R_true).trace() - 1)*0.5))
                ls_30_results.append([pos_error_ls30, ori_error_ls30])

                pos_error_ls50 = np.linalg.norm(pos_true - pos_refine50)
                R_refine50 = ToR(ori_refine50)
                R_true = ToR(ori_true)
                ori_error_ls50 = np.rad2deg(np.arccos(((R_refine50.T@R_true).trace() - 1)*0.5))
                ls_50_results.append([pos_error_ls50, ori_error_ls50])
    
        pos_mean_error_dict = {}
        ori_mean_error_dict = {}

        pos_result = []
        ori_result = []
        for data in [ hs_results, hs_10_results, hs_30_results, hs_50_results]:
            pos_err_arr = []
            ori_err_arr = []
            for i in range(len(hs_results)):
                pos_err_arr.append(data[i][0])
                ori_err_arr.append(data[i][1])
            pos_result.append(np.median(np.array(pos_err_arr)))
            ori_result.append(np.median(np.array(ori_err_arr)))
        pos_mean_error_dict['hs'] = np.array(pos_result)
        ori_mean_error_dict['hs'] = np.array(ori_result)

        pos_result = []
        ori_result = []
        for data in [ ls_results, ls_10_results,ls_30_results, ls_50_results]:
            pos_err_arr = []
            ori_err_arr = []
            for i in range(len(ls_results)):
                pos_err_arr.append(data[i][0])
                ori_err_arr.append(data[i][1])
            pos_result.append(np.median(np.array(pos_err_arr)))
            ori_result.append(np.median(np.array(ori_err_arr)))
        pos_mean_error_dict['ls'] = np.array(pos_result)
        ori_mean_error_dict['ls'] = np.array(ori_result)


        species = ("no refine", "NeFeS10", "NeFeS30", "NeFeS50")
        x = np.arange(len(species))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        plt.rc('pdf',fonttype = 42)
        plt.rc('ps',fonttype = 42)
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(layout='constrained')  
        for attribute, measurement in pos_mean_error_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            #ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.set_xticks(x + width, species)
        ax.set_ylabel('Trans. median error (m)')
        ax.set_title(f'{scene}: pose error change')
        ax.legend(loc='upper right', ncol=3)
        plt.savefig(f'../{apr}/results/bar_trans_{scene}.pdf')
        plt.clf()
        
        multiplier = 0
        plt.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(layout='constrained')  
        for attribute, measurement in ori_mean_error_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            #ax.bar_label(rects, padding=3)
            multiplier += 1
        ax.set_xticks(x + width, species)
        ax.set_ylabel('Rot. median error (deg)')
        ax.set_title(f'{scene}: pose error change')
        ax.legend(loc='upper right', ncol=3)
        plt.savefig(f'../{apr}/results/bar_rot_{scene}.pdf')
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apr", default=APRS[0], choices=APRS)
    parser.add_argument("--gamma", default=0.96)
    main(parser)
