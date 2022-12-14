import torch
import nibabel as nib


# dataloader for brain images
class ISLES_Dataset(torch.utils.data.Dataset):

    def __init__(self, imgPath,imgAnnot,cropped = False, downsample = False ,transform = None):
        super(ISLES_Dataset,self).__init__()
        # saving dataset attributes
        self.imgPath = imgPath
        self.imgAnnot = imgAnnot
        self.transform = transform
        self.cropped = cropped
        self.downsample = downsample
        self.stats = torch.Tensor([18.94762340826202, 382.20466552518104, 19.55005538419728])
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
            # optional downsampling
            if self.downsample:
                data = torch.nn.functional.interpolate(torch.unsqueeze(data,dim=0),size=(64,75,62),mode='trilinear')  
                data = torch.squeeze(data,dim=0)
            labNifti = nib.load(self.imgPath+'//'+self.imgAnnot[idx]+'//ses-1//anat//'+self.imgAnnot[idx]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')
            labArray = labNifti.get_fdata()
            label = torch.LongTensor(labArray)
            # optional downsampling
            if self.downsample:
                label = torch.unsqueeze(torch.Tensor(labArray),dim=0)
                label = torch.nn.functional.interpolate(torch.unsqueeze(label,dim=0),size=(64,75,62),mode='nearest') 
                label = torch.squeeze(torch.squeeze(label,dim=0),dim=0)
                label = label.long()
        # transforming of data - not used yet
        if self.transform:
          data = self.transform(data)
          label = self.transform(label)
        # scan standardization using calculated mean and std
        data = (data - self.stats[0])/self.stats[2]
        return data, label

# dataset for 2d brain images
class ISLES_Dataset_2D(torch.utils.data.Dataset):

    def __init__(self, imgPath,imgAnnot,sliceCount,cropped = False,downsample = False,transform = None):
        super(ISLES_Dataset_2D,self).__init__()
        # saving dataset attributes
        self.imgPath = imgPath
        self.imgAnnot = imgAnnot
        self.transform = transform
        self.cropped = cropped
        self.downsample = downsample
        self.stats = torch.Tensor([18.94734622191496, 386.08147979033464, 19.648956201038636])
        if self.cropped: sliceCount = sliceCount - 20 
        self.steps = len(self.imgAnnot)*sliceCount
        # creating annotation for loading individual slices
        self.imgSliceAnnot = []
        for i in self.imgAnnot:
            for j in range(0,sliceCount):
                self.imgSliceAnnot.append([i,j])

    def __len__(self):
        return self.steps
    
    def __getitem__(self, idx):

        if self.cropped:
            if self.downsample:
                # loading downsampled scan and annotation a extracting wanted slice
                imgNifti = nib.load(self.imgPath+'//'+self.imgSliceAnnot[idx][0]+'//ses-1//anat//'+self.imgSliceAnnot[idx][0]+'_ses-1_space-MNI152NLin2009aSym_T1w_resampfor2D.nii.gz')
                imgArray = imgNifti.get_fdata()
                imgArray = imgArray[:,:,self.imgSliceAnnot[idx][1]]
                data = torch.unsqueeze(torch.Tensor(imgArray),dim=0)  
                labNifti = nib.load(self.imgPath+'//'+self.imgSliceAnnot[idx][0]+'//ses-1//anat//'+self.imgSliceAnnot[idx][0]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask_resampfor2d.nii.gz')
                labArray = labNifti.get_fdata()
                labArray = labArray[:,:,self.imgSliceAnnot[idx][1]]
                label = torch.LongTensor(labArray)
            else:
                # loading cropped scan and annotation a extracting wanted slice
                imgNifti = nib.load(self.imgPath+'//'+self.imgSliceAnnot[idx][0]+'//ses-1//anat//'+self.imgSliceAnnot[idx][0]+'_ses-1_space-MNI152NLin2009aSym_T1w_cropped.nii.gz')
                imgArray = imgNifti.get_fdata()
                imgArray = imgArray[:,:,self.imgSliceAnnot[idx][1]]
                data = torch.unsqueeze(torch.Tensor(imgArray),dim=0)
                labNifti = nib.load(self.imgPath+'//'+self.imgSliceAnnot[idx][0]+'//ses-1//anat//'+self.imgSliceAnnot[idx][0]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask_cropped.nii.gz')
                labArray = labNifti.get_fdata()
                labArray = labArray[:,:,self.imgSliceAnnot[idx][1]]
                label = torch.LongTensor(labArray)
        else:
            # loading original scan and annotation a extracting wanted slice
            imgNifti = nib.load(self.imgPath+'//'+self.imgSliceAnnot[idx][0]+'//ses-1//anat//'+self.imgSliceAnnot[idx][0]+'_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz')
            imgArray = imgNifti.get_fdata()
            imgArray = imgArray[:,:,self.imgSliceAnnot[idx][1]]
            data = torch.unsqueeze(torch.Tensor(imgArray),dim=0)
            # optional downsampling
            if self.downsample:
                data = torch.nn.functional.interpolate(torch.unsqueeze(data,dim=0),size=(64,75),mode='bilinear')  
                data = torch.squeeze(data,dim=0)
            labNifti = nib.load(self.imgPath+'//'+self.imgSliceAnnot[idx][0]+'//ses-1//anat//'+self.imgSliceAnnot[idx][0]+'_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')
            labArray = labNifti.get_fdata()
            labArray = labArray[:,:,self.imgSliceAnnot[idx][1]]
            label = torch.LongTensor(labArray)
            # optional downsampling of annotation
            if self.downsample:
                label = torch.unsqueeze(label.float(),dim=0)
                label = torch.nn.functional.interpolate(torch.unsqueeze(label,dim=0),size=(64,75),mode='nearest') 
                label = torch.squeeze(torch.squeeze(label,dim=0),dim=0)
                label = label.long()
        # transforming of data - not used yet
        if self.transform:
          data = self.transform(data)
          label = self.transform(label)
        # data normalization using calculated mean and std
        data = (data - self.stats[0])/self.stats[2]
        return data, label