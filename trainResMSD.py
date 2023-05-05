#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% importing libraries and setting paths
import torch
import torchmetrics
import torchio as tio
import nibabel as nib
import numpy as np
import os,pickle,copy
from datetime import date
from tqdm import tqdm
from scipy import ndimage

# then import my classes
from loaders import ISLES_Dataset_MSD
from res3dunet import Res3DUNet
from loss_fcns import DiceLoss
from transforms import randomIntensity,randomContrast

# device for pytorch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# folders with dataset
imgPath = r'/storage/brno3-cerit/home/xnantl01/DPDataMSD/imagesTr'
lblPath = r'/storage/brno3-cerit/home/xnantl01/DPDataMSD/labelsTr'
predPath = r'/storage/brno3-cerit/home/xnantl01/DPDataMSD/gt_segmentations/resunetexp'

# create folder for predictions of new models
if not os.path.isdir(predPath+"/"+str(date.today())):
    os.mkdir(predPath+"/"+str(date.today()))


#%% Training using cross validation
K = 5

# loading folds from nnUNet
folds = pickle.load(open("/storage/brno3-cerit/home/xnantl01/DP/splits_final.pkl", "rb"))

# load best training parameters
# bestParams = pickle.load(open("/storage/brno3-cerit/home/xnantl01/DP/bestParam_2023-03-10.pkl","rb"))
# or expert training parameters
bestParams = pickle.load(open("/storage/brno3-cerit/home/xnantl01/DP/bestParamResExp.pkl","rb"))

for i in range(K):

    # use folds from nnUNet
    trainImg = folds[i]['train']
    valImg = folds[i]['val']
    
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
   
    tr_ds_dl = torch.utils.data.DataLoader(tr_ds,batch_size=bestParams["batchSize"].item(),shuffle=True)
    
    val_ds = ISLES_Dataset_MSD(
        imgPath = imgPath,
        lblPath = lblPath,
        imgAnnot = valImg,
        transform = None
    )
    val_ds_dl = torch.utils.data.DataLoader(val_ds,batch_size=bestParams["batchSize"].item(),shuffle=False)

    # U-Net
    net = Res3DUNet(1,2,filters=[32,64,128,256])
    net.to(device)

    # training and validation
    # loss selection
    loss_f = DiceLoss()

    net.eval()
    
    # optimizer initialization
    opt = torch.optim.Adam(net.parameters(),lr=bestParams["initLR"].item(),weight_decay=bestParams["weightDecay"].item())#5e-5

    tr_losses = []
    val_losses = []
    tr_dice_scores = []
    dice_scores = []
    best_net = []
    learning_rates = []
    best_dice = 0.0
    patience = 30
    start_epoch = 0
    end_epoch = 100

    # loading checkpoint when resuming
    # checkpoint = torch.load("./UNet3Dclassiccheckpoint1012.pt")
    # net.load_state_dict(checkpoint['state_dict'])
    # opt.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch']+ start_epoch
    # end_epoch = checkpoint['epoch'] + end_epoch
    # tr_losses = checkpoint['tr_losses']
    # val_losses = checkpoint['val_losses']
    # tr_dice_scores = checkpoint['tr_dice_scores']
    # dice_scores = checkpoint['dice_scores']

    with torch.autograd.set_detect_anomaly(True):
        for epoch in tqdm(range(start_epoch,end_epoch)):
            tr_loss = 0.0
            val_loss = 0.0
            tr_dice = 0.0
            dice = 0.0

            # LR scheduler
            opt.param_groups[0]['lr'] =  bestParams["initLR"] * (1 - epoch / end_epoch)**0.9

            # iteration through all training batches - forward pass, backpropagation, optimalization, performance evaluation
            net.train()
            print('\nEpoch {}: First training batch loading'.format(epoch))
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

            print('All training batches used')

            # iteration through all validation batches - forward pass, performance evaluation
            net.eval()
            print('\nEpoch {}: First validation batch loading'.format(epoch))
            with torch.no_grad():
                for img,lbl in val_ds_dl:
                    img,lbl = img.to(device),lbl.to(device)

                    pred=net(img)

                    loss=loss_f(pred,lbl)

                    val_loss+=loss.item()
                    pred = torch.argmax(torch.softmax(pred,dim = 1), dim = 1)
                    dice += torchmetrics.functional.dice(pred,lbl,ignore_index = 0)

            print('\nAll validation batches used')

            # eventual plotting of last validation scan every 20 epochs
            #if epoch>0 and epoch % 20 == 0:
                #fig, ax = plt.subplots(1, 2)
                #ax[0].imshow(torch.rot90(img[0,0,:,:,35].cpu()),cmap = 'gray')
                #ax[0].imshow(torch.rot90(pred[0,:,:,35].cpu()),alpha=0.5,cmap = 'copper')
                #ax[0].set_aspect(img.shape[2]/img.shape[3])
                #ax[0].set_title('Predikce')
                #ax[0].set_axis_off()
                #ax[1].imshow(torch.rot90(img[0,0,:,:,35].cpu()),cmap = 'gray')
                #ax[1].imshow(torch.rot90(lbl[0,:,:,35].cpu()),alpha=0.5,cmap = 'copper')
                #ax[1].set_aspect(img.shape[2]/img.shape[3])
                #ax[1].set_title('Zlatý standard')
                #ax[1].set_axis_off()
                #plt.show()
        
            # calculate average losses
            tr_loss=tr_loss/len(tr_ds_dl)
            val_loss=val_loss/len(val_ds_dl)
            tr_dice = tr_dice/len(tr_ds_dl)
            dice = dice/len(val_ds_dl)

            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            tr_dice_scores.append(tr_dice.detach().cpu().numpy())
            dice_scores.append(dice.detach().cpu().numpy())

            # saving best model so far
            if dice>best_dice:
                best_net = copy.deepcopy(net)
                best_dice = copy.deepcopy(dice)

            print('Fold {}; Epoch {}; LR: {}; Train Loss: {:.4f}; Valid Loss: {:.4f}; Train Dice coeff: {:.4f}; Valid Dice coeff: {:.4f}'.format(i,epoch,opt.param_groups[0]['lr'],tr_loss,val_loss,tr_dice,dice))
            # early stopping
            if (epoch > start_epoch+1 and dice_scores[-1] < dice_scores[-2]):
                patience -= 1
                if (patience == 0):
                    break
            
    # run inference and save masks
    with torch.no_grad():
        for imgId in valImg:
            # load scan
            imgNifti = nib.load(imgPath+"/"+imgId+'_0000.nii.gz')
            img = torch.Tensor(imgNifti.get_fdata())
            img = (img - torch.mean(img))/torch.std(img)
            img = torch.unsqueeze(torch.unsqueeze(img,dim=0),dim=0)
            img = img.to(device)

            # predict
            pred=net(img)
            pred = torch.argmax(torch.softmax(pred,dim = 1),dim = 1)
            pred = torch.squeeze(pred,dim=0)

            # resample to full resolution
            pred = pred.detach().cpu().numpy()
            pred = ndimage.zoom(pred,zoom=[2.76,2.84,2.73],order=0,prefilter=True)
            
            # modify header spacings
            imgHead = imgNifti.header.copy()
            zooms = np.array(imgHead.get_zooms())
            zooms = tuple(zooms/zooms)
            imgHead.set_zooms(zooms)
            
            # save prediction
            predNifti = nib.Nifti1Image(pred, imgNifti.affine, imgHead)
            nib.save(predNifti,predPath+'/'+str(date.today())+"/"+imgId+'.nii.gz')


    print('\nAll validation masks in fold {} saved'.format(i))

    # save model and metrics
    foldTraining = {
        'fold': i,
        'epoch': epoch,
        'net': net,
        'tr_losses': tr_losses,
        'val_losses': val_losses,
        'tr_dice_scores': tr_dice_scores,
        'dice_scores': dice_scores
    }
    torch.save(foldTraining, "/storage/brno3-cerit/home/xnantl01/DP/modely/ResUNetExpAugFold{}_{}.pt".format(i,date.today()))       
