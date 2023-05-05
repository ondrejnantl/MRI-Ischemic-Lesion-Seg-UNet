#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import nibabel as nib
import numpy as np
from scipy import ndimage

# string with path to predictions
predPath = r'D:/DPDataMSD/gt_segmentations/nnunet100'

# create list with names of predictions
predAnnot = sorted(os.listdir(predPath))

# iterate through subjects
for idx in range(len(predAnnot)):
    # load prediction
    predNifti = nib.load(predPath+'/'+predAnnot[idx])
    pred = predNifti.get_fdata()
    # upsample prediction
    resampPred = ndimage.zoom(pred, zoom=[2.76,2.84,2.73],order=0,prefilter=True)
    # modify prediction header
    resampPredHead = predNifti.header.copy()
    zooms = np.array(resampPredHead.get_zooms())
    zooms = tuple(zooms/zooms)
    resampPredHead.set_zooms(zooms)
    # save upsampled prediction as nifti
    resampPredNifti = nib.Nifti1Image(resampPred, predNifti.affine, resampPredHead)
    nib.save(resampPredNifti,predPath+'/'+predAnnot[idx])