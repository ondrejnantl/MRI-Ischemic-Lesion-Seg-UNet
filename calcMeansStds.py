import nibabel as nib
import numpy as np

# loading names of subjects in training dataset
trainFd= open(r'D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\data\data\trainNamesSP.txt', 'r')
trainIdx = trainFd.read()
trainIdx = trainIdx.splitlines()
trainFd.close()

# setting path to scans
imgPath = r'D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\data\data\train\derivatives\ATLAS'
#%% get mean and std for scans - modified from https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
sums = 0.0
ssums = 0.0

for i,name in enumerate(trainIdx):
    imgNifti = nib.load(imgPath+'\\'+name+'\\ses-1\\anat\\'+name+'_ses-1_space-MNI152NLin2009aSym_T1w_resamp.nii.gz')
    img = imgNifti.get_fdata()
    sums += img.sum()
    ssums += (img ** 2).sum()

count = len(trainIdx) * img.shape[0] * img.shape[1] * img.shape[2]

# mean and std
totalMean = sums / count
totalVar  = (ssums / count) - (totalMean ** 2)
totalStd  = np.sqrt(totalVar)

# save mean and std into txt file
stats= open(r'D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\data\data\stats3D.txt', 'w')
stats.write("%s\n" % totalMean)
stats.write("%s\n" % totalVar)
stats.write("%s\n" % totalStd)
stats.close()
    
#%% get mean and std for slices - modified from https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
sums = 0.0
ssums = 0.0
        
for i,name in enumerate(trainIdx):
    imgNifti = nib.load(imgPath+'\\'+name+'\\ses-1\\anat\\'+name+'_ses-1_space-MNI152NLin2009aSym_T1w_resampfor2D.nii.gz')
    img = imgNifti.get_fdata()
    sums += img.sum()
    ssums += (img ** 2).sum()

count = len(trainIdx) * img.shape[0] * img.shape[1] * img.shape[2]

# mean and std
totalMean = sums / count
totalVar  = (ssums / count) - (totalMean ** 2)
totalStd  = np.sqrt(totalVar)

# save mean and std into txt file
stats= open(r'D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\data\data\stats2D.txt', 'w')
stats.write("%s\n" % totalMean)
stats.write("%s\n" % totalVar)
stats.write("%s\n" % totalStd)
stats.close()