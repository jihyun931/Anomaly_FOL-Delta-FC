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
    num_clips = len(clips_list)

    cnt = 0
    AP = 0.0

    for i in range(num_clips):
        key_value = list(all_pred.keys())[i]
        temp_shape += len(all_pred[key_value])
        all_pred_flatten.extend(all_pred[key_value])

    min_threshold = sorted(all_pred_flatten)[0]
    max_threshold = sorted(all_pred_flatten)[-1]

    thresholds = np.arange(min_threshold, max_threshold, 0.0002)
    thresholds = np.append(thresholds, max_threshold)
    thresholds = np.delete(thresholds, 0)
    print("num of thresholds :", len(thresholds))

    Precision = np.zeros((len(thresholds)))
    Recall = np.zeros((len(thresholds)))
    Time = np.zeros((len(thresholds)))

    for threshold in thresholds:
        Tp = 0.0
        Tp_Fp = 0.0 #TP + FP
        Tp_Tn = 0.0
        frame_to_acc = 0.0
        counter = 0
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

    new_Recall = new_Recall[~np.isnan(new_Time)]
    new_Precision = new_Precision[~np.isnan(new_Time)]
    new_Time = new_Time[~np.isnan(new_Time)]

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
    plt.plot(new_Recall, new_Time/10, label='Recall-mean_time curve')
    plt.xlabel('Recall')
    plt.ylabel('TTC')
    plt.ylim([0, 10])
    plt.xlim([0.0, 1.0])
    plt.savefig(main_path + '/Recall_vs_ATTC.png')


#=============================================================
#=============================================================
main_path = "./ap_result/train_on_DoTA/concatenation/EL/"
pickle_file = main_path + "all_pred.pkl"
print(pickle_file)
with open(pickle_file, 'rb') as f:
    all_pred_original = pickle.load(f)
#=============================================================
#=============================================================


with open(pickle_file, 'rb') as f:
    all_pred = pickle.load(f)


num_clips = len(list(all_pred.keys()))
print("num of test dataset :", num_clips)

df = pd.read_excel("E:/DoTA/dataset/ego_involved/0_test/DoTA_ego_label.xlsx")
labeling_array = np.array(df)
positive_data_array = labeling_array[np.where(labeling_array[:,4] != 'o')][:,0]
print("num of positive :", len(positive_data_array))
print("num of negative :", num_clips - len(positive_data_array))

all_labels = np.zeros(num_clips)

for i in range(len(all_labels)):
    if list(all_pred.keys())[i] in positive_data_array:
        all_labels[i] = 1.0



AP, new_Time, new_Precision, new_Recall  = evaluation(all_pred, all_labels)


visualization_PR_curve(new_Precision, new_Recall, main_path)
visualization_RT_curve(new_Time, new_Recall, main_path)

ATTC = np.mean(new_Time)/10
print(">> mAP :", AP)
print(">> average time to accident:", ATTC)

harmonic_mean = 2*(AP*ATTC)/(AP+ATTC)
print(">> harmonic mean : ", harmonic_mean)