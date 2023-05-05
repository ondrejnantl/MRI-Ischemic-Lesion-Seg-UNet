#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import packages
import torch
import torchmetrics
import torchio as tio
import pickle
from datetime import date
from hyperopt import hp, tpe, fmin

# then import my classes
from loaders import ISLES_Dataset_MSD
from unet3d import UNet3D
from loss_fcns import DiceLoss
from transforms import randomIntensity,randomContrast

# training function
def run_model(params):
    print('Started another optimalization step')
    #%% model and data initialization
    # folder with dataset
    imgPath = r'/storage/brno3-cerit/home/xnantl01/DPDataMSD/imagesTr'
    lblPath = r'/storage/brno3-cerit/home/xnantl01/DPDataMSD/labelsTr'
    
    # loading folds from nnUNet
    folds = pickle.load(open("/storage/brno3-cerit/home/xnantl01/DP/splits_final.pkl", "rb"))
    
    # use folds from nnUNet
    trainImg = folds[4]['train']
    valImg = folds[4]['val']
    
    # create dataloaders
    tr_ds = ISLES_Dataset_MSD(
        imgPath = imgPath,
        lblPath = lblPath,
        imgAnnot = trainImg,
        transform = tio.Compose([
            tio.RandomAffine(scales = (0.8, 1.2),degrees = (-30, 30),isotropic = True,p = 0.2),
            tio.RandomNoise(mean = 0,std = 0.01,p = 0.15,exclude = ['a_segmentation']),
            tio.RandomBlur(std = (0.05, 0.15),p = 0.2,exclude = ['a_segmentation']),
            tio.Lambda(randomIntensity,types_to_apply = ['one_image'],p = 0.15),
            tio.Lambda(randomContrast,types_to_apply = ['one_image'],p = 0.15),
            tio.RandomGamma(log_gamma = (0.8, 1.2),p = 0.15,exclude = ['a_segmentation']),
            tio.RandomFlip(axes = (0,1,2),flip_probability = 0.5)
        ])
    )
    tr_ds_dl = torch.utils.data.DataLoader(tr_ds,batch_size=params['batchSize'],shuffle=True)
    
    val_ds = ISLES_Dataset_MSD(
        imgPath = imgPath,
        lblPath = lblPath,
        imgAnnot = valImg,
        transform = None
    )
    val_ds_dl = torch.utils.data.DataLoader(val_ds,batch_size=params['batchSize'],shuffle=False)

    # U-Net
    net = UNet3D(1,2,filters=[32,64,128,256])
    net.to(device)

    #%% training and validation
    # loss selection
    loss_f = DiceLoss()

    net.eval()
    # optimizer initialization
    opt = torch.optim.Adam(net.parameters(),lr=params['initLR'],weight_decay = params['weightDecay'])
    
    # setting number of epochs
    start_epoch = 0
    end_epoch = 10

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(start_epoch,end_epoch):
            tr_loss = 0.0
            val_loss = 0.0
            tr_dice = 0.0
            dice = 0.0

            # LR scheduler
            opt.param_groups[0]['lr'] = params['initLR'] * (1 - epoch / end_epoch)**0.9

            # iteration through all training batches - forward pass, backpropagation, optimalization, performance evaluation
            net.train()
            for img,lbl in tr_ds_dl:
                img,lbl = img.to(device),lbl.to(device)

                opt.zero_grad()
                pred = net(img)

                loss = loss_f(pred,lbl)

                loss.backward()
                opt.step()
                tr_loss+=loss.item()
                pred = torch.argmax(torch.softmax(pred,dim = 1),dim = 1)
                tr_dice += torchmetrics.functional.dice(pred,lbl,ignore_index = 0)

            # iteration through all validation batches - forward pass, performance evaluation
            net.eval()
            with torch.no_grad():
                for img,lbl in val_ds_dl:
                    img,lbl = img.to(device),lbl.to(device)

                    pred=net(img)

                    loss=loss_f(pred,lbl)

                    val_loss+=loss.item()
                    pred = torch.argmax(torch.softmax(pred,dim = 1), dim = 1)
                    dice += torchmetrics.functional.dice(pred,lbl,ignore_index = 0)

            # calculate average loss
            tr_loss=tr_loss/len(tr_ds_dl)
            val_loss=val_loss/len(val_ds_dl)
            tr_dice = tr_dice/len(tr_ds_dl)
            dice = dice/len(val_ds_dl)

            # print epoch results
            print('Fold {}; Epoch {}; LR: {}; Batch size: {}; Weight Decay: {};  Train Loss: {:.4f}; Valid Loss: {:.4f}; Train Dice coeff: {:.4f}; Valid Dice coeff: {:.4f}'.format(4,epoch,opt.param_groups[0]['lr'],params['batchSize'],params['weightDecay'],tr_loss,val_loss,tr_dice,dice))
   
    return val_loss

# device for pytorch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# setting hyperparameter search spaces
hpSpace = {
    'initLR': hp.uniform('initLR', 0.000001, 0.001),
    'weightDecay': hp.uniform('weightDecay',0.000001,0.001),
    'batchSize': hp.choice('batchSize',range(1,21))
}

# running optimalization
best = fmin(fn=run_model,
            space=hpSpace,
            algo=tpe.suggest, 
            max_evals=50)

# printing best parameters
print("Best parameters: {}".format(best))

# save the best hyperparameters
with open("/storage/brno3-cerit/home/xnantl01/DP/bestParam_{}.pkl".format(date.today()), "wb") as outfile:
    pickle.dump(best, outfile,protocol=pickle.HIGHEST_PROTOCOL)

