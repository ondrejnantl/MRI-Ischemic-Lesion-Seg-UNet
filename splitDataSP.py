import re
import os
import random
import math

# get folder content of annotated data
imgPath = r'/mnt/Data/ondrejnantl/DPData/train/derivatives/ATLAS'
imgAnnot = sorted(os.listdir(imgPath))
imgAnnot.remove(imgAnnot[0])

# defining cohort to use for training
cohorts = ["^sub-r009.*$"]
selImgAnnot = []
# finding all subject IDs from cohort
for it in cohorts:
    r = re.compile(it)
    tempList = list(filter(r.match, imgAnnot))
    selImgAnnot = selImgAnnot + tempList

# splitting subjects into training, validation and testing part
valCount = math.floor(0.15*len(selImgAnnot))
testCount = math.floor(0.25*len(selImgAnnot))
trainCount = len(selImgAnnot) - (valCount+testCount)

randIdx = random.sample(selImgAnnot,len(selImgAnnot))

valIdx = randIdx[0:(valCount)]
testIdx = randIdx[valCount:(valCount+testCount)]
trainIdx = randIdx[(valCount+testCount):len(selImgAnnot)+1]

# saving IDs in training, validation and testing part
trainFd= open(r'/mnt/Data/ondrejnantl/DPData/trainNamesSP.txt', 'w')
for item in trainIdx:
    trainFd.write("%s\n" % item)
trainFd.close()
    
valFd= open(r'/mnt/Data/ondrejnantl/DPData/valNamesSP.txt', 'w')
for item in valIdx:
    valFd.write("%s\n" % item)
valFd.close()
    
testFd= open(r'/mnt/Data/ondrejnantl/DPData/testNamesSP.txt', 'w')
for item in testIdx:
    testFd.write("%s\n" % item)
testFd.close()

print('ID lists created')
