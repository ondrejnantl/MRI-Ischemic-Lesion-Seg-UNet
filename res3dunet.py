import torch

# residual block - building block of encoder and decoder of residual 3D U-Net
class ResBlock(torch.nn.Module):
    def __init__(self, n_in,n_out):
        super(ResBlock,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        # init of residual block layers
        self.LReLU = torch.nn.LeakyReLU(negative_slope=0.02)
        self.BN1 = torch.nn.BatchNorm3d(n_in)
        self.BN2 = torch.nn.BatchNorm3d(n_in)
        self.Conv1 = torch.nn.Conv3d(n_in, n_in, 3, padding='same')
        self.Conv2 = torch.nn.Conv3d(n_in, n_out, 3, padding='same')
        self.IdenMap = torch.nn.Conv3d(n_in, n_out, 1,padding='same')
    def forward(self,x):
        # forward pass through residual part of block
        out = self.BN1(x)
        out = self.LReLU(out)
        out = self.Conv1(out)
        out = self.BN2(out)
        out = self.LReLU(out)
        out = self.Conv2(out)
        # identity mapping
        x = self.IdenMap(x)
        # addition of identity and residuum      
        out += x
        return out


class Res3DUNet(torch.nn.Module):
    def __init__(self, n_channels,n_classes,filters = [64,128,256,512]):
        super(Res3DUNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # init of modules of encoder and decoder
        self.maxpool = torch.nn.MaxPool3d(kernel_size=2)
        self.encBlock1 = ResBlock(self.n_channels, filters[0])
        self.encBlock2 = ResBlock(filters[0], filters[1])
        self.encBlock3 = ResBlock(filters[1], filters[2])
        self.encBlock4 = ResBlock(filters[2], filters[3])
        self.upSmpl1 = torch.nn.ConvTranspose3d(filters[3], filters[2], kernel_size=2,stride=2)
        self.decBlock1 = ResBlock(filters[3],filters[2])
        self.upSmpl2 = torch.nn.ConvTranspose3d(filters[2], filters[1], kernel_size=2,stride=2)
        self.decBlock2 = ResBlock(filters[2],filters[1])
        self.upSmpl3 = torch.nn.ConvTranspose3d(filters[1], filters[0], kernel_size=2,stride=2)
        self.decBlock3 = ResBlock(filters[1],filters[0])
        self.finalConv = torch.nn.Conv3d(filters[0], self.n_classes, kernel_size=1)
        
        # init of weights of Conv layers
        for i in self.modules():
            if isinstance(i, torch.nn.Conv3d) or isinstance(i, torch.nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(i.weight, mode='fan_out',nonlinearity='leaky_relu')
                
                
    def forward(self,x):
        #forward passes through encoder blocks + storing data for skip connections
        dConv1 = self.encBlock1(x)
        x = self.maxpool(dConv1)
        
        dConv2 = self.encBlock2(x)
        x = self.maxpool(dConv2)
        
        dConv3 = self.encBlock3(x)
        x = self.maxpool(dConv3)
        
        x = self.encBlock4(x)
        
        # passing through decoder blocks - upsampling, dimension matching, residual block pass
        x = self.upSmpl1(x)
        diffX = dConv3.size()[2] - x.size()[2]
        diffY = dConv3.size()[3] - x.size()[3]
        diffZ = dConv3.size()[4] - x.size()[4]
        toPad = [diffZ // 2, diffZ - diffZ // 2, diffY // 2, diffY - diffY // 2,diffX // 2, diffX - diffX // 2]
        x = torch.nn.functional.pad(x,toPad,mode = 'constant',value = 0)
        x = torch.concat([x,dConv3],1)
        x = self.decBlock1(x)
        
        x = self.upSmpl2(x)
        diffX = dConv2.size()[2] - x.size()[2]
        diffY = dConv2.size()[3] - x.size()[3]
        diffZ = dConv2.size()[4] - x.size()[4]
        toPad = [diffZ // 2, diffZ - diffZ // 2, diffY // 2, diffY - diffY // 2,diffX // 2, diffX - diffX // 2]
        x = torch.nn.functional.pad(x,toPad,mode = 'constant',value = 0)
        x = torch.concat([x,dConv2],1)
        x = self.decBlock2(x)
        
        x = self.upSmpl3(x)
        diffX = dConv1.size()[2] - x.size()[2]
        diffY = dConv1.size()[3] - x.size()[3]
        diffZ = dConv1.size()[4] - x.size()[4]
        toPad = [diffZ // 2, diffZ - diffZ // 2, diffY // 2, diffY - diffY // 2,diffX // 2, diffX - diffX // 2]
        x = torch.nn.functional.pad(x,toPad,mode = 'constant',value = 0)
        x = torch.concat([x,dConv1],1)
        x = self.decBlock3(x)
        
        # final convolution to get as many output volumes as there is classes
        x = self.finalConv(x)
        return x

      