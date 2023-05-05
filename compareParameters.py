#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% load results of models with optimized and expertly configured hyperparameters
metricsOpt = pd.read_csv("D:/andyn/OneDrive - Vysoké učení technické v Brně/diplomka/vysledky/metrics_unetcl_2023-03-15.csv",sep =";")
metricsExp = pd.read_csv("D:/andyn/OneDrive - Vysoké učení technické v Brně/diplomka/vysledky/metrics_unetclexp_2023-04-25.csv",sep =";")

# create data frames for each metric
DSC = pd.DataFrame(data=np.hstack((metricsOpt["DSC"].to_numpy().reshape(648,1),
                          metricsExp["DSC"].to_numpy().reshape(648,1))),
                   columns=["Optimalizované","Expertní"])
HD = pd.DataFrame(data=np.hstack((metricsOpt["HD"].to_numpy().reshape(648,1),
                         metricsExp["HD"].to_numpy().reshape(648,1))),
                  columns=["Optimalizované","Expertní"])
F1 = pd.DataFrame(data =np.hstack((metricsOpt["F1Score"].to_numpy().reshape(648,1),
                         metricsExp["F1Score"].to_numpy().reshape(648,1))),
                  columns=["Optimalizované","Expertní"])
LCDiff = pd.DataFrame(data=np.hstack((metricsOpt["LesionCountDiff"].to_numpy().reshape(648,1),
                                     metricsExp["LesionCountDiff"].to_numpy().reshape(648,1))),
                      columns=["Optimalizované","Expertní"])
VolDiff = pd.DataFrame(data=np.hstack((metricsOpt["VolumeDiff"].to_numpy().reshape(648,1),
                                      metricsExp["VolumeDiff"].to_numpy().reshape(648,1))),
                       columns=["Optimalizované","Expertní"])
#%% creating boxplots for comparison

plt.figure(figsize=(4,4))
sns.boxplot(DSC)
plt.ylabel("DSC [-]")
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\BoxplotDSCParam.svg",dpi=200.,bbox_inches = 'tight')

plt.figure(figsize=(4,4))
sns.boxplot(HD,showfliers = False)
plt.ylabel(r'$d_H$ [mm]')
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\BoxplotHDParam.svg",dpi=200.,bbox_inches = 'tight')

plt.figure(figsize=(4,4))
sns.boxplot(F1)
plt.ylabel(r'$F_1$ [-]')
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\BoxplotF1Param.svg",dpi=200.,bbox_inches = 'tight')

plt.figure(figsize=(4,4))
sns.boxplot(LCDiff,showfliers = False)
plt.ylabel(r'$\Delta$ n [-]')
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\BoxplotLCParam.svg",dpi=200.,bbox_inches = 'tight')

plt.figure(figsize=(4,4))
sns.boxplot(VolDiff,showfliers = False)
plt.ylabel(r'$\Delta$ V [voxely]')
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\BoxplotVolDiffParam.svg",dpi=200.,bbox_inches = 'tight')
