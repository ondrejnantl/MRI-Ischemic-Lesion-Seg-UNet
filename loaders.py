# -*- coding: utf-8 -*-
import os,sys
import torch,torchvision
import nibabel as nib
import torchio as tio

# dataloader for brain images 
class ISLES_Dataset(torch.utils.data.Dataset):

    def __init__(self, imgPath,imgAnnot,cropped = False, downsample = False,transform = None):
        super(ISLES_Dataset,self).__init__()
        # saving dataset attributes
        self.imgPath = imgPath
        self.imgAnnot = imgAnnot
        self.transform = transform
        self.cropped = cropped
        self.downsample = downsample
        self.steps = len(self.imgAnnot)

    def __len__(self):
        return self.steps
    
    def __getitem__(self, idx):
        
        if self.cropped == True:
            if self.downsample:
                # loading downsampled scan and annotation
                imgNifti = nib.load(self.imgPath+'//'+self.imgAnnot[idx]+'//ses-1//anat//'+self.imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_T1w_resamp.nii.gz')
                imgArray = imgNifti.get_fdata()
                data = torch.unsqueeze(torch.Tensor(imgArray),dim=0)  
                labNifti = nib.load(self.imgPath+'//'+self.imgAnnot[idx]+'//ses-1//anat//'+self.imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask_resamp.nii.gz')
                labArray = labNifti.get_fdata()
                label = torch.LongTensor(labArray)
            else:
                # loading cropped scan and annotation
                imgNifti = nib.load(self.imgPath+'//'+self.imgAnnot[idx]+'//ses-1//anat//'+self.imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_T1w_cropped.nii.gz')
                imgArray = imgNifti.get_fdata()
                data = torch.unsqueeze(torch.Tensor(imgArray),dim=0)  
                labNifti = nib.load(self.imgPath+'//'+self.imgAnnot[idx]+'//ses-1//anat//'+self.imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask_cropped.nii.gz')
                labArray = labNifti.get_fdata()
                label = torch.LongTensor(labArray)
        else:
            # loading original scan and annotation
            imgNifti = nib.load(self.imgPath+'//'+self.imgAnnot[idx]+'//ses-1//anat//'+self.imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz')
            imgArray = imgNifti.get_fdata()
            data = torch.unsqueeze(torch.Tensor(imgArray),dim=0)
            # optional downsampling of scan
            if self.downsample:
                data = torch.nn.functional.interpolate(torch.unsqueeze(data,dim=0),size=(64,75,62),mode='trilinear')  
                data = torch.squeeze(data,dim=0)
            labNifti = nib.load(self.imgPath+'//'+self.imgAnnot[idx]+'//ses-1//anat//'+self.imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')
            labArray = labNifti.get_fdata()
            label = torch.LongTensor(labArray)
            # optional downsampling of mask
            if self.downsample:
                label = torch.unsqueeze(torch.Tensor(labArray),dim=0)
                label = torch.nn.functional.interpolate(torch.unsqueeze(label,dim=0),size=(64,75,62),mode='nearest') 
                label = torch.squeeze(torch.squeeze(label,dim=0),dim=0)
                label = label.long()
        
        # scan standardization using scan mean and stdev
        data = (data - torch.min(data))/(torch.max(data) - torch.min(data))
        
        # transform of scan - augmentation
        if self.transform:
          sub = tio.Subject(
              one_image=tio.ScalarImage(tensor=data),
              a_segmentation=tio.LabelMap(tensor=torch.unsqueeze(label,dim=0)))
          trSub = self.transform(sub)
          data = trSub.one_image.data
          label = torch.squeeze(trSub.a_segmentation.data,dim=0)
          label = label.type(torch.LongTensor)
        
        return data, label
    
# dataloader for brain images in MSD format
class ISLES_Dataset_MSD(torch.utils.data.Dataset):

    def __init__(self, imgPath,lblPath,imgAnnot,transform = None):
        super(ISLES_Dataset_MSD,self).__init__()
        # saving dataset attributes
        self.imgPath = imgPath
        self.imgAnnot = imgAnnot
        self.lblPath = lblPath
        self.transform = transform
        self.steps = len(self.imgAnnot)

    def __len__(self):
        return self.steps
    
    def __getitem__(self, idx):
    
        # loading downsampled scan and annotation
        imgNifti = nib.load(self.imgPath+'/'+self.imgAnnot[idx]+'_0000.nii.gz')
        imgArray = imgNifti.get_fdata()
        data = torch.unsqueeze(torch.Tensor(imgArray),dim=0)  
        labNifti = nib.load(self.lblPath+'/'+self.imgAnnot[idx]+'.nii.gz')
        labArray = labNifti.get_fdata()
        label = torch.LongTensor(labArray)
    
        # scan standardization using scan mean and stdev
        data = (data - torch.mean(data))/torch.std(data)

        # transform of scan - augmentation
        if self.transform:
            sub = tio.Subject(
                one_image=tio.ScalarImage(tensor=data),
                a_segmentation=tio.LabelMap(tensor=torch.unsqueeze(label,dim=0)))
            trSub = self.transform(sub)
            data = trSub.one_image.data
            label = torch.squeeze(trSub.a_segmentation.data,dim=0)
            label = label.type(torch.LongTensor)
        
        return data, label