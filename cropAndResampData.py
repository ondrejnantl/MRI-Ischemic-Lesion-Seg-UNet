# -*- coding: utf-8 -*-
import nibabel as nib
import os
import numpy as np
from scipy import ndimage

# %% loading subject folders names
imgPath = r'D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\data\data\train\derivatives\ATLAS'
imgAnnot = sorted(os.listdir(imgPath))
imgAnnot.remove(imgAnnot[0])
# %% crop and save data
for idx in range(len(imgAnnot)):
    # load scan and annotation
    imgNifti = nib.load(imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz')
    img = imgNifti.get_fdata() 
    lblNifti = nib.load(imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')
    lbl = lblNifti.get_fdata()
    # crop scan and mask from all sides by 10 voxels
    cropImg = img[10:-10,10:-10,10:-10]
    cropLbl = lbl[10:-10,10:-10,10:-10]
    # create nifti objects with modified data
    cropImgNifti = nib.Nifti1Image(cropImg, imgNifti.affine, imgNifti.header)
    cropLblNifti = nib.Nifti1Image(cropLbl, lblNifti.affine, lblNifti.header)
    # save scan and mask
    nib.save(cropImgNifti,imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_T1w_cropped.nii.gz')
    nib.save(cropLblNifti,imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask_cropped.nii.gz')

# %% crop, resample and save data
for idx in range(len(imgAnnot)):
    # load scan and annotation
    imgNifti = nib.load(imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz')
    img = imgNifti.get_fdata() 
    lblNifti = nib.load(imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')
    lbl = lblNifti.get_fdata()
    # crop scan and mask from all sides by 10 voxels
    cropImg = img[10:-10,10:-10,10:-10]
    cropLbl = lbl[10:-10,10:-10,10:-10]
    # resample data to shape 64 x 75 x 62
    resampImg = ndimage.zoom(cropImg,zoom=[0.36,0.35,0.365],order=1,prefilter=True)
    resampLbl = ndimage.zoom(cropLbl,zoom=[0.36,0.35,0.365],order=0,prefilter=True)
    # copy headers and modify spacing
    imgHead = imgNifti.header.copy()
    lblHead = lblNifti.header.copy()
    voxelZoom = (cropImg.shape[0]/resampImg.shape[0],cropImg.shape[1]/resampImg.shape[1],cropImg.shape[2]/resampImg.shape[2]) 
    imgHead.set_zooms(voxelZoom)
    lblHead.set_zooms(voxelZoom)
    # create nifti objects with modified data
    resampImgNifti = nib.Nifti1Image(resampImg, imgNifti.affine, imgHead)
    resampLblNifti = nib.Nifti1Image(resampLbl, lblNifti.affine, lblHead)
    # save scan and mask
    nib.save(resampImgNifti,imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_T1w_resamp.nii.gz')
    nib.save(resampLblNifti,imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask_resamp.nii.gz')
# %% crop, resample in x and y and save data
for idx in range(len(imgAnnot)):
    # load scan and annotation
    imgNifti = nib.load(imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz')
    img = imgNifti.get_fdata() 
    lblNifti = nib.load(imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')
    lbl = lblNifti.get_fdata()
    # crop scan and mask from all sides by 10 voxels
    cropImg = img[10:-10,10:-10,10:-10]
    cropLbl = lbl[10:-10,10:-10,10:-10]
    # resample data to shape 64 x 75 x 189 - only in plane downsampling
    resampImg = np.zeros((64,75,cropImg.shape[2]))
    resampLbl = np.zeros((64,75,cropLbl.shape[2]))
    for i in range(0,cropImg.shape[2]):
        resampImg[:,:,i] = ndimage.zoom(cropImg[:,:,i],zoom=[0.36,0.35],order=1,prefilter=True)
        resampLbl[:,:,i] = ndimage.zoom(cropLbl[:,:,i],zoom=[0.36,0.35],order=0,prefilter=True)
    # copy headers and modify spacing
    imgHead = imgNifti.header.copy()
    lblHead = lblNifti.header.copy()
    voxelZoom = (cropImg.shape[0]/resampImg.shape[0],cropImg.shape[1]/resampImg.shape[1],cropImg.shape[2]/resampImg.shape[2]) 
    imgHead.set_zooms(voxelZoom)
    lblHead.set_zooms(voxelZoom)
    # create nifti objects with modified data
    resampImgNifti = nib.Nifti1Image(resampImg, imgNifti.affine, imgHead)
    resampLblNifti = nib.Nifti1Image(resampLbl, lblNifti.affine, lblHead)
    # save scan and mask
    nib.save(resampImgNifti,imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_T1w_resampfor2D.nii.gz')
    nib.save(resampLblNifti,imgPath+'\\'+imgAnnot[idx]+'\\ses-1\\anat\\'+imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask_resampfor2d.nii.gz')        