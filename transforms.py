#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import random

# function for random intensity transform
def randomIntensity(inputScan: torch.Tensor) -> torch.Tensor:
    multiVals = (0.9,1.2)
    # multiplying scan by random scalar
    outputScan = inputScan * random.uniform(multiVals[0],multiVals[1])
    return outputScan

# function for random contrast transform
def randomContrast(inputScan: torch.Tensor) -> torch.Tensor:
    multiVals = (0.85,1.3)
    low = torch.min(inputScan)
    high = torch.max(inputScan)
    # multiplying scan by random scalar and cropping data to original dynamic range
    outputScan = inputScan * random.uniform(multiVals[0],multiVals[1])
    outputScan[outputScan<low] = low
    outputScan[outputScan>high] = high
    return outputScan