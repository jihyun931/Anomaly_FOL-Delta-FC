import sys
import os 
import numpy as np
import time
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils import data

from lib.models.rnn_ed_LEA import FolRNNED
from lib.utils.fol_dataloader_absolute import HEVIDataset_wo_ego
from config.config import * 

from lib.utils.data_prep_utils import bbox_denormalize, cxcywh_to_x1y1x2y2
from lib.utils.eval_utils import compute_IOU

print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_fol(fol_model, test_gen):
    '''
    Validate future vehicle localization module 
    Params:
        fol_model: The fol model as nn.Module
        test_gen: test data generator
    Returns:
        
    '''
    fol_model.eval() # Sets the module in training mode.


    fol_loss = 0

    loader = tqdm(test_gen, total=len(test_gen))

    FDE = 0
    ADE = 0
    FIOU = 0
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion = data

            # run forward
            fol_predictions,_ = fol_model(input_bbox, input_flow)
            
            # convert to numpy array, use [0] since batchsize if 1 for test
            fol_predictions = fol_predictions.to('cpu').numpy()[0] 
            input_bbox = input_bbox.to('cpu').numpy()[0]
            target_bbox = target_bbox.to('cpu').numpy()[0]

            # compute FDE, ADE and FIOU metrics used in FVL2019ICRA paper
            input_bbox = np.expand_dims(input_bbox, axis=1)

            input_bbox = bbox_denormalize(input_bbox, W=1280, H=640)
            fol_predictions = bbox_denormalize(fol_predictions, W=1280, H=640)
            target_bbox = bbox_denormalize(target_bbox, W=1280, H=640)
            
            fol_predictions_xyxy = cxcywh_to_x1y1x2y2(fol_predictions)
            target_bbox_xyxy = cxcywh_to_x1y1x2y2(target_bbox)

            ADE += np.mean(np.sqrt(np.sum((target_bbox_xyxy[:,:,:2] - fol_predictions_xyxy[:,:,:2]) ** 2, axis=-1)))
            FDE += np.mean(np.sqrt(np.sum((target_bbox_xyxy[:,-1,:2] - fol_predictions_xyxy[:,-1,:2]) ** 2, axis=-1)))
            tmp_FIOU = []
            for i in range(target_bbox_xyxy.shape[0]):
                tmp_FIOU.append(compute_IOU(target_bbox_xyxy[i,-1,:], fol_predictions_xyxy[i,-1,:], format='x1y1x2y2'))
            FIOU += np.mean(tmp_FIOU)
    print("FDE: %4f;    ADE: %4f;   FIOU: %4f" % (FDE, ADE, FIOU))
    ADE /= len(test_gen.dataset)
    FDE /= len(test_gen.dataset)
    FIOU /= len(test_gen.dataset)
    print("FDE: %4f;    ADE: %4f;   FIOU: %4f" % (FDE, ADE, FIOU))

def main(args):
    # initialize model
    print("Initializing pre-trained fol network")
    fol_model = FolRNNED_wo_ego(args).to(device)
    fol_model.load_state_dict(torch.load(args.best_fol_model))


    # initialize datasets
    print("Initializing test dataset...")
    dataloader_params ={
            "batch_size": args.batch_size,
            "shuffle": args.shuffle,
            "num_workers": args.num_workers
        }

    test_set = HEVIDataset_wo_ego(args, 'val')
    print("Number of test samples:", test_set.__len__())
    test_gen = data.DataLoader(test_set, **dataloader_params)

    input_bbox, input_flow, input_ego_motion, target_bbox, target_ego_motion = test_set.__getitem__(1)
    print("input shape: ", input_bbox.shape)
    print("target shape: ", target_bbox.shape)


    test_fol(fol_model, test_gen)


if __name__=='__main__':
    # load args
    args = parse_args()
    main(args)
