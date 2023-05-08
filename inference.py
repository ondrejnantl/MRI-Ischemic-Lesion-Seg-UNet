#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import os
import nibabel as nib
import numpy as np
import argparse
from unet3d import UNet3D

def inference(pathToData,pathToNets,outPath,device):
    # inputs
    # pathToData - path to data for inference
    # pathToNets - path to models for running inference - all in one folder
    # outPath - path to folder for storing predictions
    # device - device for running inference - cpu or cuda
    
    # getting the list of scans for inference
    scans = sorted(os.listdir(pathToData))

    # getting the list of trained nets
    nets = sorted(os.listdir(pathToNets))

    # iterating through all scans
    for i in range(len(scans)):
        # load scan
        imgNifti = nib.load(pathToData+"/"+scans[i])
        img = torch.Tensor(imgNifti.get_fdata())
        img = (img - torch.mean(img))/torch.std(img)
        img = torch.unsqueeze(torch.unsqueeze(img,dim=0),dim=0)
        img = img.to(device)
        # creating output
        allPred = np.zeros(img.shape[2:6])
        # iterating through all nets
        for j in range(len(nets)):
            #loading fold data and extracting the trained net
            foldData = torch.load(pathToNets+"/"+nets[j])
            net = foldData["net"]
            net.to(device)
            net.eval()
            # prediction
            with torch.no_grad():
                pred = net(img)    
                pred = torch.argmax(torch.softmax(pred,dim = 1),dim = 1)
                pred = torch.squeeze(pred,dim=0)
                pred = pred.detach().cpu().numpy()  
            # adding prediction    
            allPred += pred
        # majority vote for voxel classification - segmentation    
        allPred = 1*(allPred>=(len(nets) // 2))
        allPredNifti = nib.Nifti1Image(allPred,imgNifti.affine,imgNifti.header)
        nib.save(allPredNifti,outPath+"/"+scans[i])

if __name__ == '__main__':
    # create parser for input arguments
    parser = argparse.ArgumentParser(prog='inference',description='Run inference of stroke lesion segmentation model using trained models (excluding nnUNet models)')
    parser.add_argument('-pd',help='path to the data for inference (all shall be in this folder, no subfolders)',dest="pathToData",required=True)
    parser.add_argument('-pn',help='path to the nets for inference (all shall be in this folder, no subfolders)',dest="pathToNets",required=True)
    parser.add_argument('-po',help='path to store the predictions',dest="outPath",required=True)
    parser.add_argument('-d',choices=["cpu","cuda"],default='cpu',help='device to use for inference',dest="torchDevice",required=True)

    # parse input arguments
    args = parser.parse_args()

    # device for pytorch
    device = args.torchDevice

    # setting path to data
    pathToData = args.pathToData

    # setting path to trained nets
    pathToNets = args.pathToNets

    # setting the path for predictions
    outPath = args.outPath

    # calling inference function
    inference(pathToData,pathToNets,outPath,device)    


