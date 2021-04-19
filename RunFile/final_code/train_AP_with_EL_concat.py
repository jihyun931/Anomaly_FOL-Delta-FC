#AP :  Accident Prediction
#

import sys
import os 
import numpy as np
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchsummaryX import summary

from lib.utils.ap_train_val_utils_EL import train_ap_wo_ego, val_ap_wo_ego
from lib.models.ap_model_LEA import AP_wo_ego
from lib.utils.ap_dataloader_for_LEA import load_fol_hidden_state
from config.config import * 

from tensorboardX import SummaryWriter
import pandas as pd


# GPU connection
print("Cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load args
args = parse_args()

if args.enc_concat_type == 'cat':
    args.dec_hidden_size = args.box_enc_size + args.flow_enc_size
else:
    if args.box_enc_size != args.flow_enc_size:
        raise ValueError('Box encoder size %d != flow encoder size %d'
                            %(args.box_enc_size,args.flow_enc_size))
    else:
        args.dec_hidden_size = args.box_enc_size

print(">> Setting the Accident Precition model ... ")
AP_model = AP_wo_ego(args).to(device)
all_params = AP_model.parameters()
#optimizer = optim.RMSprop(all_params, lr=args.lr)
optimizer = optim.Adam(all_params, lr=args.lr)

dataloader_params ={
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
        "num_workers": args.num_workers
    }

val_set = load_fol_hidden_state(args, 'val')
print(">> Number of validation samples:", val_set.__len__())
val_gen = data.DataLoader(val_set, **dataloader_params)


print(">> Check the Model's architecture")
summary(AP_model, 
        torch.zeros(1, args.segment_len, args.pred_timesteps, args.dec_hidden_size).to(device)
        )


print(">> Train data root:", args.data_root)

writer = SummaryWriter('summary/train_on_DoTA/concatenation/EL/')


# MODEL TRAINING
min_loss = 1e6

best_ap_model = None

#save train(mAP, ATTC), val(mAP, ATTC)
inform = np.zeros((args.train_epoch, 4))

for epoch in range(1, args.train_epoch+1):
    print("\n")
    print("=====================================")
    print("// Epoch :", epoch)
    # regenerate the training dataset 
    train_set = load_fol_hidden_state(args, 'train')
    train_gen = data.DataLoader(train_set, **dataloader_params)
    print("Number of training samples:", train_set.__len__())

    start = time.time()

    #===== train
    train_loss, train_mAP, train_ATTC   = train_ap_wo_ego(epoch, AP_model, optimizer, train_gen, verbose=True)
    writer.add_scalar('data/train_loss', train_loss, epoch)
    writer.add_scalar('data/train_mAP', train_mAP, epoch)
    writer.add_scalar('data/train_ATTC', train_ATTC, epoch)
    inform[epoch-1,0] = train_mAP
    inform[epoch-1,1] = train_ATTC
    # print('====> Epoch: {} object pred loss: {:.4f}'.format(epoch, train_loss))

    #===== evaluation
    val_loss, val_mAP, val_ATTC = val_ap_wo_ego(epoch, AP_model, val_gen, verbose=True)
    writer.add_scalar('data/val_loss', val_loss, epoch)
    writer.add_scalar('data/val_mAP', val_mAP, epoch)
    writer.add_scalar('data/val_ATTC', val_ATTC, epoch)
    inform[epoch-1,2] = val_mAP
    inform[epoch-1,3] = val_ATTC


    # print time
    elipse = time.time() - start
    print("Elipse: ", elipse)

    # save checkpoint per epoch
    saved_ap_model_name = 'epoch_' + str(format(epoch,'03')) + '.pt'
    print("Saving checkpoints: " + saved_ap_model_name)
    torch.save(AP_model.state_dict(), os.path.join(args.checkpoint_dir, saved_ap_model_name))

df = pd.DataFrame(inform)
df.to_csv(args.checkpoint_dir+"/train_val_inform.csv", index=False)
np.save(args.checkpoint_dir+"/train_val_inform" ,inform)