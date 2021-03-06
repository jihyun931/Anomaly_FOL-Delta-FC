#AP :  Accident Prediction
# Evaluation Accident Anticiaption
# Input : pickle file (information - bounding box, optical flow, frame id


import sys
import os 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
from sklearn import metrics
import pandas as pd

import torch
from torch.utils import data

from lib.models.rnn_ed_LEA import FolRNNED_wo_ego
from lib.models.ap_model_LEA import AP_wo_ego

from config.config import * 


# GPU connection
print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load args
args = parse_args()

def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print("Error: Creating directory." + directory)

def precondition(dataset_name):
    # except dataset that is accident classification error
    # except dataset that is not properly "detection & tracking".
    df = pd.read_excel("E:/DoTA/dataset/ego_involved/0_test/DoTA_ego_label.xlsx")
    labeling_array = np.array(df)
    
    index = np.where(labeling_array[:,0] == dataset_name)[0][0]
    if labeling_array[index,4] == 'o':
        print("this test dataset except !!!")
        except_bool = True
        frame_end = 0
    else:
        except_bool = False
        frame_end = labeling_array[index,1]

    return except_bool, int(frame_end)

def cal_single_obj_risk_score(ap_result, frame_end):
    if len(ap_result) ==0:
        risk_score = np.zeros((frame_end+1,))
    else:
        risk_score = np.zeros((frame_end+1,))
        obj_frame_id = ap_result[:,1]
        obj_frame_id = obj_frame_id.astype(np.int64)
        obj_risk_score = ap_result[:,0]
        for index, id in enumerate(obj_frame_id):
            risk_score[id] = obj_risk_score[index]
    return risk_score


def cal_overall_risk_score(objs_ap_results, frame_end):
    if len(objs_ap_results) ==0:
        overall_risk_score = np.zeros((frame_end+1,))
    else:
        num_objs = len(objs_ap_results)
        objs_risk_scores = np.zeros((num_objs, frame_end+1))
        #print("objs_risk_scores:", objs_risk_scores.shape)

        for k in range(num_objs):
            obj_ap_result = objs_ap_results[k]
            obj_frame_id = obj_ap_result[:,1]
            obj_frame_id = obj_frame_id.astype(np.int64)
            obj_risk_score = obj_ap_result[:,0]

            for index, id in enumerate(obj_frame_id):
                objs_risk_scores[k,id] = obj_risk_score[index]

        overall_risk_score = np.max(objs_risk_scores, axis=0) #select max risk scores among objects
    return overall_risk_score

def vis_obj_risk_score_graph(ap_result, frame_end, clip_name, vis_save_path):
    # visualization object's predicition risk scores and ground truth scores graph.
    # save the graph.
    pickle_name = pickle_file.split(".")[0]
    risk_score = np.zeros((frame_end+1,))
    obj_frame_id = ap_result[:,1]
    obj_frame_id = obj_frame_id.astype(np.int64)
    obj_risk_score = ap_result[:,0]
    for index, id in enumerate(obj_frame_id):
        risk_score[id] = obj_risk_score[index]

    time = np.array(range(frame_end+1))
    plt.clf()
    plt.ylim(-0.1,1.1)
    plt.xlim(0, frame_end)
    plt.ylabel('risk score')
    plt.xlabel('time(frame)')
    plt.plot(time, risk_score, 'r-')
    plt.savefig(vis_save_path+ clip_name +"/"+ pickle_name + '.png')

def vis_risk_score_graph(overall_risk_score, frame_end, vis_save_path, graph_name):
    # visualization predicition risk scores and ground truth scores graph.
    # save the graph.
    time = np.array(range(frame_end+1))
    plt.clf()
    plt.ylim(-0.1,1.1)
    plt.xlim(0, frame_end)
    plt.ylabel('risk score')
    plt.xlabel('time(frame)')
    plt.title(graph_name, loc='center')
    plt.plot(time, overall_risk_score, 'r-')
    plt.savefig(vis_save_path+ clip_name + '.png')


def accident_prediction(args, test_file, fol_model, ap_model, acc_frame_id):
    #use AP model.
    #risk scores.
    with open(test_file, 'rb') as f:
        feature_data = pickle.load(f) #['bbox'] ['flow'] ['frame_id']
    
    pred_risk_scores =[] #list
    
    frame_id = feature_data['frame_id']
    frame_id = np.array(frame_id)
    limit_index = np.where(frame_id <= acc_frame_id)[0][-1]

    frame_id = frame_id[0:limit_index+1]
    bbox = feature_data['bbox'][0:limit_index+1]
    flow = feature_data['flow'][0:limit_index+1]

    total_length = frame_id.shape[0]
    box_h = torch.zeros(1, args.box_enc_size).to(device)
    flow_h = torch.zeros(1, args.flow_enc_size).to(device)

    for i in range(total_length):
        input_box = torch.from_numpy(bbox[i]).float().to(device)
        input_flow = torch.from_numpy(flow[i]).float().to(device)
        input_box = input_box.view(1,input_box.size()[0]) #[1,4]
        input_flow = input_flow.view(1, input_flow.size()[0], input_flow.size()[1], input_flow.size()[2]) #[1,5,5,2]

        risk_score, box_h, flow_h = ap_model.predict(input_box, input_flow, box_h, flow_h, fol_model)
        risk_score = risk_score.cpu().detach().numpy() #tensor to numpy array [1,1]
        risk_score = risk_score[0][0]
        pred_risk_scores.append(risk_score)

    pred_risk_scores = np.array(pred_risk_scores)
    return pred_risk_scores, frame_id



# load pre-trained trajectory prediction model
print(">> Load pre-trained FOL model ... ")
# 1) load model structure
fol_model = FolRNNED_wo_ego(args).to(device)
# 2) load model weigths
fol_model.load_state_dict(torch.load(args.best_fol_model))
fol_model.eval()

# load pre-trained accident ancitipation model
print(">> Load pre-trained Accident Precition model ... ")
AP_model = AP_wo_ego(args).to(device)
AP_model.load_state_dict(torch.load(args.best_ap_model))
AP_model.eval()


#===================================================================
# Need to Set UP.
#===================================================================
test_dataset_path = "./test_dataset/DoTA_test_dataset"
save_path = "./ap_result/train_on_DoTA/average/EL/"
createFolder(save_path)
print(">> ap_model :", args.best_ap_model)
print(">> Save path :", save_path)

vis_save_path = save_path+"visualization/"
createFolder(vis_save_path)
#===================================================================
#===================================================================

annotation_file_path = "E:/DoTA/dataset/annotations/"
df = pd.read_excel("E:/DoTA/dataset/ego_involved/0_test/DoTA_ego_label.xlsx")
labeling_array = np.array(df)
positive_data_array = labeling_array[:,0]

all_obj_risk_score ={}
all_dataset_risk_score = {}

test_dataset_list = os.listdir(test_dataset_path)
for clip_name in test_dataset_list:
    print("-- Test clip name:", clip_name)
    createFolder(vis_save_path+ clip_name)
    
    if clip_name in positive_data_array:
        print(" positive data ")
        except_bool, frame_end = precondition(clip_name)
        graph_name = "positive_data"
        if except_bool == True:
            continue

    else :
        print(" negative data ")
        graph_name = "negative_data"
        annotation_file = annotation_file_path + clip_name +".json"
        with open(annotation_file) as json_file:
            json_data = json.load(json_file)
            frame_end = json_data["anomaly_start"] - 1
        if frame_end == 0:
            continue

    print(frame_end)

    pickle_file_list = os.listdir(test_dataset_path+"/" + clip_name)
    #print(" the number of detected objects:", len(pickle_file_list))

    objs_ap_results =[]

    for pickle_file in pickle_file_list:
        test_file = test_dataset_path + "/" + clip_name + "/" + pickle_file

        with open(test_file, 'rb') as f:
            feature_data = pickle.load(f) #['bbox'] ['flow'] ['frame_id']
        frame_id = feature_data['frame_id']
        if frame_id[0] >= frame_end:
            continue

        
        pred_risk_scores, frame_id = accident_prediction(args, test_file, fol_model, AP_model, frame_end)
        pred_risk_scores = np.reshape(pred_risk_scores, (-1,1))
        frame_id = np.reshape(frame_id, (-1,1))

        ap_result = np.concatenate((pred_risk_scores, frame_id), axis=1)
        #plot accident probability graph of obejct
        vis_obj_risk_score_graph(ap_result, frame_end, clip_name, vis_save_path)
        

        single_obj_risk_score = cal_single_obj_risk_score(ap_result, frame_end)
        pickle_name = pickle_file.split(".")[0]
        all_obj_risk_score[pickle_name] = single_obj_risk_score

        objs_ap_results.append(ap_result)

    #calculate video-clip's overall prediction risk score
    overall_risk_score = cal_overall_risk_score(objs_ap_results, frame_end)
    print("clip's risk score shape :", overall_risk_score.shape)
    #plot accident probability graph of video clip
    vis_risk_score_graph(overall_risk_score, frame_end, vis_save_path, graph_name)

    all_dataset_risk_score[clip_name] = overall_risk_score


#save pickle file

# 1) save the accident probability of each object
result_file = save_path +"/all_pred.pkl"
with open(result_file, 'wb') as f:
    pickle.dump(all_dataset_risk_score, f, pickle.HIGHEST_PROTOCOL)

# 2) save the accident probability of each video clip
obj_result_file = save_path +"/all_obj_pred.pkl"
with open(obj_result_file, 'wb') as f:
    pickle.dump(all_obj_risk_score, f, pickle.HIGHEST_PROTOCOL)


