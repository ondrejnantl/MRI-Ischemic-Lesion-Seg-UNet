# import numpy
import torch
import torch.nn as nn

# class for loss combining cross entropy and Dice loss - Dice loss is inspired by: https://github.com/CoinCheung/pytorch-loss/blob/master/dice_loss.py
class CrossEntropyDiceLoss(nn.Module):
    def __init__(self,weights = [0.5,0.5]):
        super(CrossEntropyDiceLoss,self).__init__()
        CrossEntropyDiceLoss.CE = torch.nn.CrossEntropyLoss(weight = torch.Tensor(weights).to("cuda"))
    def forward(self,pred,lbl):
        
        # calculating cross entropy using PyTorch module
        CrossEnt = CrossEntropyDiceLoss.CE(pred,lbl)
        
        #softmax activation layer
        pred = torch.softmax(pred,dim=1)
        
        # GT to one hot encoding
        lbl_extend=lbl.clone()
        lbl_extend.unsqueeze_(1)
        one_hot = torch.cuda.FloatTensor(lbl_extend.size(0), 2, lbl_extend.size(2), lbl_extend.size(3),lbl_extend.size(4)).zero_()
        one_hot.scatter_(1, lbl_extend, 1) 
        
        TotalDice = 0.0
        # calculating DSC for every class as average DSC in batch
        for i in range(one_hot.shape[1]):
            predT = pred[:,i].contiguous().view(pred.shape[0],-1)
            one_hotT = one_hot[:,i].contiguous().view(one_hot.shape[0],-1)
            num = 2 * torch.mul(predT,one_hotT).sum(dim = 1) + 1
            den = predT.pow(2).sum(dim=1) + one_hotT.pow(2).sum(dim=1) + 1
                                      
            Dice = 1 - num/den
            TotalDice += Dice.mean()
       
        return CrossEnt+TotalDice/one_hot.shape[1]

# class for Dice loss - inspired by: https://github.com/CoinCheung/pytorch-loss/blob/master/dice_loss.py
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
       
    def forward(self, pred, lbl):
        
        # softmax activation layer      
        pred = torch.softmax(pred,dim=1)

        # GT to one hot encoding
        lbl_extend=lbl.clone()
        lbl_extend.unsqueeze_(1) 
        one_hot = torch.cuda.FloatTensor(lbl_extend.size(0), 2, lbl_extend.size(2), lbl_extend.size(3),lbl_extend.size(4)).zero_()
        one_hot.scatter_(1, lbl_extend, 1)
        
        # calculating DSC for every class as average DSC of class in batch
        TotalDice = 0.0
        for i in range(one_hot.shape[1]):
            predT = pred[:,i].contiguous().view(pred.shape[0],-1)
            one_hotT = one_hot[:,i].contiguous().view(one_hot.shape[0],-1)
            num = 2 * torch.mul(predT,one_hotT).sum(dim = 1) + 1
            den = predT.pow(2).sum(dim=1) + one_hotT.pow(2).sum(dim=1) + 1
                                       
            Dice = 1 - num/den
            TotalDice += Dice.mean()
        
        # calculating average DSC
        return TotalDice/one_hot.shape[1]

# class for 2D Dice loss - inspired by: https://github.com/CoinCheung/pytorch-loss/blob/master/dice_loss.py    
class DiceLoss2D(nn.Module):
    def __init__(self):
        super(DiceLoss2D, self).__init__()
       
    def forward(self, pred, lbl):
        
        # softmax activation layer      
        pred = torch.softmax(pred,dim=1)

        # GT to one hot encoding
        lbl_extend=lbl.clone()
        lbl_extend.unsqueeze_(1) 
        one_hot = torch.cuda.FloatTensor(lbl_extend.size(0), 2, lbl_extend.size(2), lbl_extend.size(3)).zero_()
        one_hot.scatter_(1, lbl_extend, 1)
        
        # calculating DSC for every class as average DSC of class in batch
        TotalDice = 0.0
        for i in range(one_hot.shape[1]):
            predT = pred[:,i].contiguous().view(pred.shape[0],-1)
            one_hotT = one_hot[:,i].contiguous().view(one_hot.shape[0],-1)
            num = 2 * torch.mul(predT,one_hotT).sum(dim = 1) + 1
            den = predT.pow(2).sum(dim=1) + one_hotT.pow(2).sum(dim=1) + 1
                                       
            Dice = 1 - num/den
            TotalDice += Dice.mean()
        
        # calculating average DSC
        return TotalDice/one_hot.shape[1]

# Tversky loss was taken from https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5, alpha=0.5, beta=0.5):
        
        # softmax activation layer
        inputs = torch.softmax(inputs,dim = 1)

        # GT to one hot encoding
        targets_extend=targets.clone()
        targets_extend.unsqueeze_(1) 
        one_hot = torch.cuda.FloatTensor(targets_extend.size(0), 2, targets_extend.size(2), targets_extend.size(3),targets_extend.size(4)).zero_()
        one_hot.scatter_(1, targets_extend, 1)       
        
        # calculating FPs and FNs for each batch sample
        dims = (1, 2, 3, 4)
        intersection = torch.sum(inputs * one_hot, dims)
        fps = torch.sum(inputs * (-one_hot + 1.0), dims)
        fns = torch.sum((-inputs + 1.0) * one_hot, dims)

        numerator = intersection
        denominator = intersection + alpha * fps + beta * fns
        tversky_loss = numerator / (denominator + smooth)

        return torch.mean(-tversky_loss + 1.0)