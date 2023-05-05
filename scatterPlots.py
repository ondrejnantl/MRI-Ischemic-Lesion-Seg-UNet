# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

#%% load results for all 3 methods
metricsNn = pd.read_csv("D:/andyn/OneDrive - Vysoké učení technické v Brně/diplomka/vysledky/metrics_nnunet_2023-03-15.csv",sep =";")
metricsCl = pd.read_csv("D:/andyn/OneDrive - Vysoké učení technické v Brně/diplomka/vysledky/metrics_unetcl_2023-03-15.csv",sep =";")
metricsRes = pd.read_csv("D:/andyn/OneDrive - Vysoké učení technické v Brně/diplomka/vysledky/metrics_resunetrand_2023-03-25.csv",sep =";")
# calculate nonparametric correlation
corrNn = stats.spearmanr(a = metricsNn["GTVolume"].to_numpy(),b = metricsNn["PredVolume"].to_numpy())
corrCl = stats.spearmanr(a = metricsCl["GTVolume"].to_numpy(),b = metricsCl["PredVolume"].to_numpy())
corrRes = stats.spearmanr(a = metricsRes["GTVolume"].to_numpy(),b = metricsRes["PredVolume"].to_numpy())

#%% plot DSC scatter for all methods
plt.figure(figsize=(7,2))
sns.scatterplot(data = metricsCl,x="GTVolume",y="DSC")
plt.xscale('log')
plt.ylim(0.,1.)
plt.xlabel("Objem léze [voxely]")
plt.ylabel("DSC [-]")
plt.title("Závislost DSC na objemu léze - UNet")
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\ScatterplotCl.svg",dpi=200.,bbox_inches = 'tight')


plt.figure(figsize=(7,2))
sns.scatterplot(data = metricsRes,x="GTVolume",y="DSC")
plt.xscale('log')
plt.ylim(0.,1.)
plt.xlabel("Objem léze [voxely]")
plt.ylabel("DSC [-]")
plt.title("Závislost DSC na objemu léze - ResUNet")
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\ScatterplotRes.svg",dpi=200.,bbox_inches = 'tight')


plt.figure(figsize=(7,2))
sns.scatterplot(data = metricsNn,x="GTVolume",y="DSC")
plt.xscale('log')
plt.ylim(0.,1.)
plt.xlabel("Objem léze [voxely]")
plt.ylabel("DSC [-]")
plt.title("Závislost DSC na objemu léze - nnUNet")
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\ScatterplotNn.svg",dpi=200.,bbox_inches = 'tight')
#%% plot volumes for all methods
plt.figure(figsize=(7,2))
sns.regplot(data = metricsCl,x="GTVolume",y="PredVolume",fit_reg=True)
plt.text(0, 500000, "spearmanr: "+str(corrCl[0])+" ; p = "+str(corrCl[1]), horizontalalignment='left', size='small', color='black')
plt.xlabel("Objem léze v anotaci [voxely]")
plt.ylabel("Objem léze v predikci [voxely]")
plt.title("Závislost objemu léze v predikci na objemu léze v anotaci - UNet")
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\VolScatterplotCl.svg",dpi=200.,bbox_inches = 'tight')


plt.figure(figsize=(7,2))
sns.regplot(data = metricsRes,x="GTVolume",y="PredVolume",fit_reg=True)
plt.text(0, 450000, "spearmanr: "+str(corrRes[0])+" ; p = "+str(corrRes[1]), horizontalalignment='left', size='small', color='black')
plt.xlabel("Objem léze v anotaci [voxely]")
plt.ylabel("Objem léze v predikci [voxely]")
plt.title("Závislost objemu léze v predikci na objemu léze v anotaci - ResUNet")
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\VolScatterplotRes.svg",dpi=200.,bbox_inches = 'tight')

plt.figure(figsize=(7,2))
sns.regplot(data = metricsNn,x="GTVolume",y="PredVolume",fit_reg=True)
plt.text(0, 450000, "spearmanr: "+str(corrNn[0])+" ; p = "+str(corrNn[1]), horizontalalignment='left', size='small', color='black')
plt.xlabel("Objem léze v anotaci [voxely]")
plt.ylabel("Objem léze v predikci [voxely]")
plt.title("Závislost objemu léze v predikci na objemu léze v anotaci - nnUNet")
plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\vysledky\VolScatterplotNn.svg",dpi=200.,bbox_inches = 'tight')