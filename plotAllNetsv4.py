# -*- coding: utf-8 -*-

import nibabel as nib
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from datetime import date

# set path to annotation and all predictions
imgPath = r'D:/DPDataMSD/imagesTrFR'
lblPath = r'D:/DPDataMSD/labelsTrFR'
unetPath = r'D:/DPDataMSD/gt_segmentations/unetcl'
resunetPath = r'D:/DPDataMSD/gt_segmentations/resunet'
nnunetPath = r'D:/DPDataMSD/gt_segmentations/nnunet100'
#%% load file list - MSD
imgAnnot = sorted(os.listdir(lblPath))
#%%  load and show 5 subjects - image and annot masks 

subIdx = np.array([[320,75,53,505,137],
                   [235,623,467,543,203],
                   [549,512,256,95,147]]) #544 

# for 3 examples with 5 subjects each
for subsI in range(subIdx.shape[0]):

    fig, ax = plt.subplots(subIdx.shape[1], 4,figsize=(4, subIdx.shape[1]))      
    
    # for all subjects in example
    for i in range(subIdx.shape[1]):
        # load cropped image and mask
        imgNifti = nib.load(imgPath+'/'+imgAnnot[subIdx[subsI,i]][0:-7]+'_0000.nii.gz')
        img = imgNifti.get_fdata() 
        lblNifti = nib.load(lblPath+'/'+imgAnnot[subIdx[subsI,i]])
        lbl = lblNifti.get_fdata()
        centroid = np.floor(np.mean(np.argwhere(lbl),axis=0)).astype(np.int64)
        
        # load classic net mask
        predClNifti = nib.load(unetPath+'/'+imgAnnot[subIdx[subsI,i]])
        predCl = predClNifti.get_fdata()
        
        # load resunet mask
        predResNifti = nib.load(resunetPath+'/'+imgAnnot[subIdx[subsI,i]])
        predRes = predResNifti.get_fdata()
        
        # load nnunet mask
        predNnNifti = nib.load(nnunetPath+'/'+imgAnnot[subIdx[subsI,i]])
        predNn = predNnNifti.get_fdata()
        predNn[predNn>1.0] = 1.0
        
        # plotting

        ax[i,0].imshow(np.rot90(img[:,:,centroid[2]]),cmap = 'gray')
        ax[i,0].set_aspect(img.shape[0]/img.shape[1])
        ax[0,0].set_title('Originál')
        ax[i,0].set_axis_off()
        
        ax[i,1].imshow(np.rot90(img[:,:,centroid[2]]),cmap = 'gray')
        ax[i,1].imshow(np.rot90(np.dstack((predCl[:,:,centroid[2]],lbl[:,:,centroid[2]],predCl[:,:,centroid[2]]))),alpha=0.5)
        ax[i,1].set_aspect(img.shape[0]/img.shape[1])
        ax[0,1].set_title('UNet')
        ax[i,1].set_axis_off()
        
        ax[i,2].imshow(np.rot90(img[:,:,centroid[2]]),cmap = 'gray')
        ax[i,2].imshow(np.rot90(np.dstack((predRes[:,:,centroid[2]],lbl[:,:,centroid[2]],predRes[:,:,centroid[2]]))),alpha=0.5)
        ax[i,2].set_aspect(img.shape[0]/img.shape[1])
        ax[0,2].set_title('ResUNet')
        ax[i,2].set_axis_off()
        
        ax[i,3].imshow(np.rot90(img[:,:,centroid[2]]),cmap = 'gray')
        ax[i,3].imshow(np.rot90(np.dstack((predNn[:,:,centroid[2]],lbl[:,:,centroid[2]],predNn[:,:,centroid[2]]))),alpha=0.5)
        ax[i,3].set_aspect(img.shape[0]/img.shape[1])
        ax[0,3].set_title('nnUNet')
        ax[i,3].set_axis_off()
        
    plt.tight_layout(pad = 0.5)
        
    # saving
    fig.savefig(r"D:/andyn/OneDrive - Vysoké učení technické v Brně/diplomka/obrazky/vysledky/plotResultsv2Stack{}_{}.png".format(subsI,date.today()),dpi=200.,bbox_inches = 'tight')
    




