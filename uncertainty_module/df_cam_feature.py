import argparse
import numpy as np
from numpy.linalg import norm
from utils.functions import *

SCENES = ['KingsCollege','OldHospital','ShopFacade','StMarysChurch']
APRS = ['DFNet', "MS-Transformer", "PoseNet"]
def main(parser):
    args = parser.parse_args()
    apr = args.apr
    scene = args.scene
    feature_path = f'datasets/Cambridge/{scene}/'
    gt_file = f'../{apr}/Cambridge/{scene}_train_gt.txt'
    test_gt_file = f'../{apr}/Cambridge/{scene}_test_gt.txt'
    pred_file = f'../{apr}/Cambridge/{scene}_predict.txt'
    dist_th = 1.5
    dist_str = str(dist_th).replace('0.','') + '0'
    test_simi_file = f'../{apr}/simi_ranking/{scene}_simi_{dist_str}.txt'
    test_list = get_test_file(test_gt_file)
    simi_dict = {}
    time_total = 0
    for img in test_list:
        pos_out, ori_out = getPredictPos(img,pred_file)
        pos_out = np.array(pos_out)
        ori_out = ori_out / np.sqrt(np.sum(ori_out**2))
        img_retrieved, delta_time = retrievalImg(pos_out,gt_file,dist_th)
        time_total += delta_time
        simi_topk = []
        if len(img_retrieved) == 0:
            simi_dict[img] = 0
        else:
            for img_r in img_retrieved:
                imgQ_feature = get_feature(feature_path,img)
                imgR_feature = get_feature(feature_path,img_r)
                similarity = np.dot(imgQ_feature, imgR_feature.T)/(norm(imgQ_feature)*norm(imgR_feature))
                simi_topk.append(similarity[0][0])
            simi_dict[img] = max(simi_topk)

    simi_list = sorted(simi_dict.items(), key = lambda x:x[1])
    print_simi_list(simi_list,test_simi_file)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apr", default=APRS[0], choices=APRS)
    parser.add_argument("--scene", default=SCENES[0], choices=SCENES)
    main(parser)

