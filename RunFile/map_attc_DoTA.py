#Calculate Mean Average Precision & Average Time To Collision

import sys
import os 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def evaluation(all_pred, all_labels):
    temp_shape = 0
    all_pred_flatten = []
    clips_list = list(all_pred.keys())

    for i in range(num_clips):
        key_value = list(all_pred.keys())[i]
        temp_shape += len(all_pred[key_value])
        all_pred_flatten.extend(all_pred[key_value])

    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0

    for threshold in sorted(all_pred_flatten):
        Tp = 0.0
        Tp_Fp = 0.0 #TP + FP
        Tp_Tn = 0.0
        frame_to_acc = 0.0
        counter = 0.0
        for i in range(len(clips_list)):
            tp =  np.where(all_pred[clips_list[i]]*all_labels[i]>=threshold)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                frame_to_acc += len(all_pred[clips_list[i]]) - tp[0][0]
                counter = counter+1
            Tp_Fp += float(len(np.where(all_pred[clips_list[i]]>=threshold)[0])>0)
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            Time[cnt] = np.nan
        else:
            Time[cnt] = (frame_to_acc/counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _,rep_index = np.unique(Recall,return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]
    new_Recall = new_Recall[:-1]
    new_Precision = new_Precision[:-1]
    new_Time = new_Time[:-1]

    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    return AP, new_Time, new_Precision, new_Recall

def visualization_PR_curve(new_Precision, new_Recall, main_path):
    plt.clf()
    plt.plot(new_Recall, new_Precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(main_path + '/PR_curve.png')

def visualization_RT_curve(new_Time, new_Recall, main_path):
    plt.clf()
    plt.plot(new_Recall, new_Time, label='Recall-mean_time curve')
    plt.xlabel('Recall')
    plt.ylabel('frame')
    plt.ylim([0, 100])
    plt.xlim([0.0, 1.0])
    plt.title('Recall-mean_time' )
    plt.savefig(main_path + '/Recall_vs_ATTC.png')


# main_path = "./ap_result/train_on_DoTA/p_200_n_100_layer_2/"
main_path = "./ap_result/bbox_flow_pos_neg_test_deltaAda2/"
pickle_file = main_path + "all_pred.pkl"

with open(pickle_file, 'rb') as f:
    all_pred = pickle.load(f)
#del all_pred['4OGV0AbV91U_000667'] #anomaly start = 1

# If you want to except Detection Error dataset
#-------------------------------------------------------------------------------
#df = pd.read_excel("D:/DoTA/dataset/ego_involved/0_test/DoTA_ego_label.xlsx")
#labeling_array = np.array(df)
#dataset_name = labeling_array[:,0]
#detection_error = np.where(labeling_array[:,3] == 'o')[0]
#for i in detection_error:
#    clip_name = dataset_name[i]
#    del all_pred[clip_name]
#---------------------------------------------------------------------------------


num_clips = len(list(all_pred.keys()))
print("num of test dataset :", num_clips)

# df = pd.read_excel("D:/DoTA/dataset/ego_involved/0_test/DoTA_ego_label.xlsx")
# df = pd.read_excel("./test_dataset/0_test/Intern_Kor_label.xlsx")
df = pd.read_excel("./test_dataset/0_test/DoTA_ego_label.xlsx")
labeling_array = np.array(df)
# positive_data_array = labeling_array[np.where(labeling_array[:,4] != 'o')][:,0]
positive_data_array = labeling_array[np.where(np.logical_and(labeling_array[:,4] != 'o', labeling_array[:,3] != 'o'))][:,0]
print("num of positive :", len(positive_data_array))
print("num of negative :", num_clips - len(positive_data_array))

all_labels = np.zeros(num_clips)

for i in range(len(all_labels)):
    if list(all_pred.keys())[i] in positive_data_array:
        all_labels[i] = 1.0



AP, new_Time, new_Precision, new_Recall  = evaluation(all_pred, all_labels)
visualization_PR_curve(new_Precision, new_Recall, main_path)
visualization_RT_curve(new_Time, new_Recall, main_path)
print(">> mAP :", AP)
new_Time = np.nan_to_num(new_Time, copy=True, nan=0.0, posinf=None, neginf=None)
print(">> average time to accident:", np.mean(new_Time)/10)