#Train trjaectory prediciton model.
#[reference] https://github.com/MoonBlvd/tad-IROS2019

import sys
import os 
import numpy as np
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchsummaryX import summary

from lib.utils.train_val_utils import  train_fol, val_fol
from lib.models.rnn_ed_LEA import FolRNNED_wo_ego
from lib.utils.fol_dataloader_absolute import HEVIDataset_wo_ego
from config.config import * 

from tensorboardX import SummaryWriter

# GPU connection
print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load args from config file
args = parse_args()

# initialize model
fol_model = FolRNNED_wo_ego(args).to(device)
all_params = fol_model.parameters()

# Optimizer
optimizer = optim.RMSprop(all_params, lr=args.lr)

# initialize datasets
print("Initializing train and val datasets...")
dataloader_params ={
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
        "num_workers": args.num_workers
    }
val_set = HEVIDataset_wo_ego(args, 'val')
print("Number of validation samples:", val_set.__len__())
val_gen = data.DataLoader(val_set, **dataloader_params)

# print model architecture
summary(fol_model, 
        torch.zeros(1, args.segment_len, 4).to(device),
        torch.zeros(1, args.segment_len, 50).to(device))


# summary writer
# summary storage location
writer = SummaryWriter('summary/fol_vanilla_wo_ego/HEVI/seg_16_pred_10_absolute/concatenation/')


# Train
all_val_loss = []
min_loss = 1e6
best_fol_model = None
best_ego_model = None

for epoch in range(1, args.nb_fol_epoch+1):

    train_set = HEVIDataset_wo_ego(args, 'train')
    train_gen = data.DataLoader(train_set, **dataloader_params)
    print("Number of training samples:", train_set.__len__())

    start = time.time()
    # train
    train_loss = train_fol(epoch, fol_model, optimizer, train_gen)
    writer.add_scalar('data/train_loss', train_loss, epoch)

    # val
    val_loss = val_fol(epoch, fol_model, val_gen)
    writer.add_scalar('data/val_loss', val_loss, epoch)
    all_val_loss.append(val_loss)

    # print processing time
    elipse = time.time() - start
    print("Elipse: ", elipse)

    # save checkpoints if loss decreases
    if val_loss < min_loss:
        try:
            os.remove(best_fol_model)
            os.remove(best_ego_model)
        except:
            pass

        min_loss = val_loss
        saved_fol_model_name = 'fol_epoch_' + str(format(epoch,'03')) + '_loss_%.4f'%val_loss + '.pt'
        

        print("Saving checkpoints: " + saved_fol_model_name)
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)

        torch.save(fol_model.state_dict(), os.path.join(args.checkpoint_dir, saved_fol_model_name))


        best_fol_model = os.path.join(args.checkpoint_dir, saved_fol_model_name)
