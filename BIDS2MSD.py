#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import nibabel
import shutil
from nnunet.dataset_conversion.utils import generate_dataset_json

#%% loading IDs of all annotated downsampled data
imgPath = r'/mnt/Data/ondrejnantl/DPData/train/derivatives/ATLAS'
imgAnnot = sorted(os.listdir(imgPath))
imgAnnot.remove(imgAnnot[0])

#%% copying downsampled training scans and masks
for idx in range(0,len(imgAnnot)):
    # copy scan
    source = imgPath+'/'+imgAnnot[idx]+'/ses-1/anat/'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_T1w_resamp.nii.gz'
    if (idx+1) < 10:
        target = '/home/ondrejnantl/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Task501_ATLASResamp/imagesTr/ATLASResamp_00'+str(idx+1)+'_0000.nii.gz'
    elif (idx+1) < 100:
        target = '/home/ondrejnantl/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Task501_ATLASResamp/imagesTr/ATLASResamp_0'+str(idx+1)+'_0000.nii.gz'
    elif (idx+1) >= 100:
        target = '/home/ondrejnantl/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Task501_ATLASResamp/imagesTr/ATLASResamp_'+str(idx+1)+'_0000.nii.gz'
    shutil.copy(source,target)
    
    # copy mask
    source = imgPath+'/'+imgAnnot[idx]+'/ses-1/anat/'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask_resamp.nii.gz'
    if (idx+1) < 10:
        target = '/home/ondrejnantl/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Task501_ATLASResamp/labelsTr/ATLASResamp_00'+str(idx+1)+'.nii.gz'
    elif (idx+1) < 100:
        target = '/home/ondrejnantl/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Task501_ATLASResamp/labelsTr/ATLASResamp_0'+str(idx+1)+'.nii.gz'
    elif (idx+1) >= 100:
        target = '/home/ondrejnantl/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Task501_ATLASResamp/labelsTr/ATLASResamp_'+str(idx+1)+'.nii.gz'
    shutil.copy(source,target)

#%% create json file for running U-Net, nnUNet shall be properly installed

generate_dataset_json("/home/ondrejnantl/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Task501_ATLASResamp/dataset.json", 
                      "/home/ondrejnantl/nnUNetFrame/dataset/nnUNet_raw/nnUNet_raw_data/Task501_ATLASResamp/imagesTr", 
                      None, 
                      ("T1"), 
                      {0: "background", 1: "lesion"}, 
                      "ATLAS R2.0 - resampled",
                      dataset_description="This is a resampled version of ATLAS R2.0. It contains T1W images of brain after stroke.",
                      dataset_reference="https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html") 