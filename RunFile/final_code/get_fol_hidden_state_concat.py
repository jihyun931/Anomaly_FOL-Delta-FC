import sys
import os 
import numpy as np
import time
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils import data
from torchsummaryX import summary


from lib.models.rnn_ed_LEA import FolRNNED_wo_ego
from lib.utils.ap_dataloader_for_LEA import ei_accident_dataset_no_padding
from config.config import * 

import pickle

# GPU connection
print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_fol_output(args, fol_model, dataset_gen, save_path):
    fol_model.eval()

    loader = tqdm(dataset_gen, total=len(dataset_gen))

    n = 0

    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            n += 1
            result = {}
             
            input_bbox, input_flow, input_ego_motion, target_risk_score = data

            batch_size = input_bbox.size()[0]
            length = input_bbox.size()[1]

            if args.enc_concat_type == 'cat':
                args.dec_hidden_size = args.box_enc_size + args.flow_enc_size
            else:
                if args.box_enc_size != args.flow_enc_size:
                    raise ValueError('Box encoder size %d != flow encoder size %d'
                                        %(args.box_enc_size,args.flow_enc_size))
                else:
                    args.dec_hidden_size = args.box_enc_size

            # Initial hidden state : zeros
            box_h = torch.zeros(batch_size, args.box_enc_size).to(device)
            flow_h = torch.zeros(batch_size, args.flow_enc_size).to(device)

            list_all_dec_h = torch.zeros(length, args.pred_timesteps, args.dec_hidden_size).to(device)

            for i in range(length):
                box = input_bbox[:,i,:]  #box : [batch_size, 4]
                flow = input_flow[:,i,:] #flow : [batch_size, 5,5,2]
                predicts, box_h, flow_h, all_dec_h = fol_model.predict(box, flow, box_h, flow_h)
                list_all_dec_h[i,:,:] = all_dec_h[0,:,:]

            #print("list_all_dec_h:", list_all_dec_h.shape) #if concat [length, predic_timesteps, 1024]
            save_array = list_all_dec_h.cpu().detach().numpy()
            target_risk_score = target_risk_score.cpu().detach().numpy()
            

            result['hidden_state'] = save_array #if concat [length, predic_timesteps, 1024]
            result['target_risk_score'] = target_risk_score[0]

            save_file = save_path + "{:04}".format(n) + ".pkl"
            
            with open(save_file, 'wb') as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


    return


def main(args):
    # initialize model
    if args.with_ego:
        fol_model = FolRNNED(args).to(device)
        fol_model.load_state_dict(torch.load(args.best_fol_model))
    else :
        fol_model = FolRNNED_wo_ego(args).to(device)
        fol_model.load_state_dict(torch.load(args.best_fol_model))

    # initialize datasets
    print("Initializing dataset...")

    #=========================
    args.batch_size = 1
    #=========================

    dataloader_params ={
            "batch_size": args.batch_size,
            "shuffle": args.shuffle,
            "num_workers": args.num_workers
        }

    dataset = ei_accident_dataset_no_padding(args, 'train')
    save_path = "./ap_train_dataset/DoTA_based/semi_output/hidden_state/absolute_fol_concat/train/"


    print(">> Number of dataset:", dataset.__len__())
    dataset_gen = data.DataLoader(dataset, **dataloader_params)

    input_bbox, input_flow, input_ego_motion, target_risk_score = dataset.__getitem__(1)
    print(" -- input_bbox: ", input_bbox.shape) #[length, 4]
    print(" -- input_flow: ", input_flow.shape) #[length, 5,5,2]
    print(" -- target_risk_score: ", target_risk_score.shape) #[1]

    get_fol_output(args, fol_model, dataset_gen, save_path)


if __name__=='__main__':
    # load args
    args = parse_args()
    main(args)
