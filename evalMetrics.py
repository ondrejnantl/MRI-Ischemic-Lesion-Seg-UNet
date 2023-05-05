# -*- coding: utf-8 -*-
import os
import nibabel as nib
import numpy as np
import pandas as pd
import atlasScoring # this is code from ATLAS R2.0 - Stroke Lesion Segmentation Challenge
import skimage

# setting path to annotation and predictions(same names for each subjects annotation and prediction) - change these
predPath = r'D:/DPDataMSD/gt_segmentations/resunetexp'
lblPath = r'D:/DPDataMSD/labelsTrFR'

# get list of predictions in folder
predAnnot = sorted(os.listdir(predPath))

# prealocating variables for metrics
DSC = np.zeros((648))
Hausdorff = np.ones((648))
F1Score = np.zeros((648))
LesionCountDiff = np.zeros((648))
VolumeDiff = np.zeros((648))
GTVolume = np.zeros((648))
PredVolume = np.zeros((648))

# iterate through subjects
for idx in range(0,648):
    # load prediction
    predNifti = nib.load(predPath+'/'+predAnnot[idx])
    pred = predNifti.get_fdata()
    pred = np.expand_dims(pred,0)
    # load annotation
    lblNifti = nib.load(lblPath+'/'+predAnnot[idx])
    lbl = lblNifti.get_fdata()
    lbl = np.expand_dims(lbl,0)
    
    # calculate metrics
    DSC[idx] = atlasScoring.dice_coef(lbl, pred)
    Hausdorff[idx] = skimage.metrics.hausdorff_distance(lbl,pred)
    F1Score[idx] = atlasScoring.lesion_f1_score(lbl, pred)
    LesionCountDiff[idx] = atlasScoring.simple_lesion_count_difference(lbl, pred)
    VolumeDiff[idx] = atlasScoring.volume_difference(lbl, pred)
    GTVolume[idx] = np.sum(lbl)
    PredVolume[idx] = np.sum(pred)
    
# create summary table of metrics for all subjects
metrics = pd.DataFrame(
    data = np.hstack(
        (DSC.reshape(648,1),
         Hausdorff.reshape(648,1),
         F1Score.reshape(648,1),
         LesionCountDiff.reshape(648,1),
         VolumeDiff.reshape(648,1),
         GTVolume.reshape(648,1),
         PredVolume.reshape(648,1))),
    columns= ["DSC","HD","F1Score","LesionCountDiff","VolumeDiff","GTVolume","PredVolume"])

# save it in csv
metrics.to_csv("D:/andyn/OneDrive - Vysoké učení technické v Brně/diplomka/vysledky/metrics_resunetexp_2023-04-25.csv",index=False,sep=";")