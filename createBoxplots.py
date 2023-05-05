# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% load summary tables
metricsNn = pd.read_csv("D:/andyn/OneDrive - Vysoké učení technické v Brně/diplomka/vysledky/metrics_nnunet100_2023-04-22.csv",sep =";")
metricsCl = pd.read_csv("D:/andyn/OneDrive - Vysoké učení technické v Brně/diplomka/vysledky/metrics_unetcl_2023-03-15.csv",sep =";")
metricsRes = pd.read_csv("D:/andyn/OneDrive - Vysoké učení technické v Brně/diplomka/vysledky/metrics_resunetrand_2023-03-25.csv",sep =";")

# create data frames for each metric
DSC = pd.DataFrame(data=np.hstack((metricsCl["DSC"].to_numpy().reshape(648,1),
                          metricsRes["DSC"].to_numpy().reshape(648,1),
                          metricsNn["DSC"].to_numpy().reshape(648,1))),
                   columns=["Standardní","Residuální","nnUNet"])
HD = pd.DataFrame(data=np.hstack((metricsCl["HD"].to_numpy().reshape(648,1),
                         metricsRes["HD"].to_numpy().reshape(648,1),
                         metricsNn["HD"].to_numpy().reshape(648,1))),
                  columns=["Standardní","Residuální","nnUNet"])
F1 = pd.DataFrame(data =np.hstack((metricsCl["F1Score"].to_numpy().reshape(648,1),
                         metricsRes["F1Score"].to_numpy().reshape(648,1),
                         metricsNn["F1Score"].to_numpy().reshape(648,1))),
                  columns=["Standardní","Residuální","nnUNet"])
LCDiff = pd.DataFrame(data=np.hstack((metricsCl["LesionCountDiff"].to_numpy().reshape(648,1),
                                     metricsRes["LesionCountDiff"].to_numpy().reshape(648,1),
                                     metricsNn["LesionCountDiff"].to_numpy().reshape(648,1))),
                      columns=["Standardní","Residuální","nnUNet"])
VolDiff = pd.DataFrame(data=np.hstack((metricsCl["VolumeDiff"].to_numpy().reshape(648,1),
                                      metricsRes["VolumeDiff"].to_numpy().reshape(648,1),
                                      metricsNn["VolumeDiff"].to_numpy().reshape(648,1))),
                       columns=["Standardní","Residuální","nnUNet"])
GTVolume = pd.DataFrame(data=np.hstack((metricsCl["GTVolume"].to_numpy().reshape(648,1),
                                      metricsRes["GTVolume"].to_numpy().reshape(648,1),
                                      metricsNn["GTVolume"].to_numpy().reshape(648,1))),
                       columns=["Standardní","Residuální","nnUNet"])
PredVolume = pd.DataFrame(data=np.hstack((metricsCl["PredVolume"].to_numpy().reshape(648,1),
                                      metricsRes["PredVolume"].to_numpy().reshape(648,1),
                                      metricsNn["PredVolume"].to_numpy().reshape(648,1))),
                       columns=["Standardní","Residuální","nnUNet"])
#%% plotting boxplots for all metrics
plt.figure(figsize=(4,4))
sns.boxplot(DSC) #,showfliers = False
plt.ylabel("DSC [-]")
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\BoxplotDSC.svg",dpi=200.,bbox_inches = 'tight')

plt.figure(figsize=(4,4))
sns.boxplot(HD,showfliers = False)
plt.ylabel(r'$d_H$ [mm]')
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\BoxplotHD.svg",dpi=200.,bbox_inches = 'tight')

plt.figure(figsize=(4,4))
sns.boxplot(F1) #,showfliers = False
plt.ylabel(r'$F_1$ [-]')
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\BoxplotF1.svg",dpi=200.,bbox_inches = 'tight')

plt.figure(figsize=(4,4))
sns.boxplot(LCDiff,showfliers = False)
plt.ylabel(r'$\Delta$ n [-]')
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\BoxplotLCDiff.svg",dpi=200.,bbox_inches = 'tight')

plt.figure(figsize=(4,4))
sns.boxplot(VolDiff,showfliers = False)
plt.ylabel(r'$\Delta$ V [voxely]')
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\BoxplotVolDiff.svg",dpi=200.,bbox_inches = 'tight')
