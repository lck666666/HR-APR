import numpy as np
import time

def ToR(q):
    return np.eye(3) + 2 * np.array((
        (-q[2] * q[2] - q[3] * q[3], q[1] * q[2] -
         q[3] * q[0], q[1] * q[3] + q[2] * q[0]),
        (q[1] * q[2] + q[3] * q[0], -q[1] * q[1] -
         q[3] * q[3], q[2] * q[3] - q[1] * q[0]),
        (q[1] * q[3] - q[2] * q[0],
         q[2] * q[3] + q[1] * q[0],
         -q[1] * q[1] - q[2] * q[2])))

def getPredictPos(img_name,pred_file):
    text = []
    with open(pred_file) as f:
        for line in f.readlines():
            data = line.split('/t/n')
            for str in data:
                sub_str = str.split(' ')
                text.append(sub_str)
    num_database = len(text)
    #print(text)
    for i in range(num_database):
        if img_name in text[i]:
            pos_predict = [0,0,0]
            ori_predict = [0,0,0,0]
            for j in range(3):
                pos_predict[j] = float(text[i][j+1])
            for k in range(4):
                ori_predict[k] = float(text[i][k+4])
    pos_predict = np.array(pos_predict)
    ori_predict = np.array(ori_predict)
    ori_predict = ori_predict / np.sqrt(np.sum(ori_predict**2))
    return pos_predict,ori_predict

def getPos(img_path,test_gt_file):
    text = []
    with open(test_gt_file) as f:
            for line in f.readlines():
                data = line.split('\n')
                for stri in data:
                    sub_str = stri.split(' ')
                    text.append(sub_str)
    for i in range(len(text)):
            if img_path in text[i]:
                for j in range(len(text[i])):
                    if (j > 0) and text[i][j] != '':
                        text[i][j] = float(text[i][j])
                pos_true = text[i][1:4]
                quat_true = text[i][4:8]
                break
    pos_true = np.array(pos_true)
    ori_true = np.array(quat_true)
    ori_true = ori_true / np.sqrt(np.sum(ori_true**2))
    return pos_true,ori_true

def get_test_file(test_gt_file):
    text = []
    with open(test_gt_file) as f:
        for line in f.readlines()[3:]:
            data = line.split('\n')
            for stri in data:
                if stri != '':
                    sub_str = stri.split(' ')
                    text.append(sub_str[0])
    return text

def print_simi_list(simi_list, path):
    with open(path, 'w') as f:
        for line in simi_list:
            f.write(line[0] + ' ')
            f.write(str(line[1]))
            f.write('\n')

def getSimi(img_path, file_path):
    text = []
    with open(file_path) as f:
            for line in f.readlines():
                data = line.split('\n')
                for stri in data:
                    sub_str = stri.split(' ')
                    text.append(sub_str)
    for i in range(len(text)):
            if img_path in text[i]:
                text[i][1] = float(text[i][1])
                simi_score = text[i][1]
                break
    return simi_score

def retrievalImg(pos_out,gt_file,dist_th):
    pose_next = False
    text = []
    with open(gt_file) as f:
        for line in f.readlines()[3:]:
            data = line.split('/t/n')
            for str in data:
                sub_str = str.split(' ')
                text.append(sub_str)
    num_database = len(text)
    closest_images = []
    retrieval_time = 0
    for i in range(num_database):
        new_pos_database = [0,0,0]
        new_ori_database = [0,0,0,0]
        for j in range(3):
            new_pos_database[j] = float(text[i][j+1])
        for k in range(4):
            new_ori_database[k] = float(text[i][k+4])
        pos_r =  np.array(new_pos_database)
        time_start = time.time()
        dist = np.linalg.norm(pos_out - pos_r)
        time_end = time.time()
        if dist <= dist_th:
            closest_images.append(text[i][0])
            pose_next = True
        retrieval_time += time_end - time_start
    if (pose_next):
        return closest_images, retrieval_time
    else:
        return [],retrieval_time

def get_feature(feature_path, img_path):
    path = feature_path + img_path.replace('png','npy')
    feature_np = np.load(path)
    return feature_np