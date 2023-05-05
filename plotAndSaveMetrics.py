import torch
import numpy as np
import matplotlib.pyplot as plt
#%% plot data from all folds for one model
for i in range(0,5):
    # load fold data
    foldData = torch.load(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\modely\ResUNetAugFold{}_2023-03-24.pt".format(i),map_location=torch.device("cpu"))
    # create figure
    fig,ax = plt.subplots()
    # plot losses
    ax.plot(foldData['tr_losses'],color='blue')
    ax.plot(foldData['val_losses'],color='red')
    ax.set_xlabel("Epochy")
    ax.set_ylabel("Kriteriální funkce")
    # plot DSC
    ax2=ax.twinx()
    ax2.plot(foldData['tr_dice_scores'],color='orange')
    ax2.plot(foldData['dice_scores'],color='green')
    fig.legend(['tr_loss','val_loss','tr_dice','val_dice'],loc = 'upper right', bbox_to_anchor=(0.875, 0.875),ncol = 2)
    ax2.set_ylabel("DSC")
    ax2.set_ylim(0, 1)
    plt.title('Kriteriální funkce a DSC - Reziduální 3D UNet - Fold {}'.format(i))
    # save figure
    plt.savefig(r"D:\andyn\OneDrive - Vysoké učení technické v Brně\diplomka\obrazky\trenink\TrainingPlotResUNetRandFold{}_2023-03-15.svg".format(i),dpi=200.,bbox_inches = 'tight')

