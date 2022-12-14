import torch

# function that returns convolutional block 
# with the specific number of input and output feature maps
def tandemConv(n_in,n_out):
    layers = torch.nn.Sequential(
        torch.nn.Conv3d(n_in,n_out,kernel_size=3,padding=1,bias=False),
        torch.nn.BatchNorm3d(n_out),
        torch.nn.LeakyReLU(negative_slope=0.02),
        torch.nn.Conv3d(n_out,n_out,kernel_size=3,padding=1,bias=False),
        torch.nn.BatchNorm3d(n_out),
        torch.nn.LeakyReLU(negative_slope=0.02)
    )
    return layers

# function that returns the upsampling layer
def upConv(n_in,n_out):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose3d(n_in,n_out,kernel_size=2,stride=2),
    )
    

class UNet3D(torch.nn.Module):

    def __init__(self,channels,classes,filters = [32,64,128,256]):
        super().__init__()

        # init of modules
        self.downC1 = tandemConv(channels,filters[0])
        self.downC2 = tandemConv(filters[0],filters[1])
        self.downC3 = tandemConv(filters[1],filters[2])
        self.downC4 = tandemConv(filters[2],filters[3])
        
        self.maxpool = torch.nn.MaxPool3d(2,2)
        
        self.upC1U = upConv(filters[3],filters[2])
        self.upC1T = tandemConv(filters[3],filters[2])
        self.upC2U = upConv(filters[2],filters[1])
        self.upC2T = tandemConv(filters[2],filters[1])
        self.upC3U = upConv(filters[1],filters[0])
        self.upC3T = tandemConv(filters[1],filters[0])

        self.classConv = torch.nn.Conv3d(filters[0],classes,kernel_size=1)

        
        # init of weights of Conv layers
        for i in self.modules():
            if isinstance(i, torch.nn.Conv3d) or isinstance(i, torch.nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(i.weight, mode='fan_out',nonlinearity='leaky_relu')

    def forward(self,x):
        #forward passes through encoder blocks + storing data for skip connections
        dConv1 = self.downC1(x)
        x = self.maxpool(dConv1)

        dConv2 = self.downC2(x)
        x = self.maxpool(dConv2)

        dConv3 = self.downC3(x)
        x = self.maxpool(dConv3)

        x = self.downC4(x)

        # passing through decoder blocks - upsampling, dimension matching, convolutional block pass
        x = self.upC1U(x)
        diffX = dConv3.size()[2] - x.size()[2]
        diffY = dConv3.size()[3] - x.size()[3]
        diffZ = dConv3.size()[4] - x.size()[4]
        toPad = [diffZ // 2, diffZ - diffZ // 2, diffY // 2, diffY - diffY // 2,diffX // 2, diffX - diffX // 2]
        x = torch.nn.functional.pad(x,toPad,mode = 'constant',value = 0)
        x = torch.concat([x,dConv3],1)
        x = self.upC1T(x)

        x = self.upC2U(x)
        diffX = dConv2.size()[2] - x.size()[2]
        diffY = dConv2.size()[3] - x.size()[3]
        diffZ = dConv2.size()[4] - x.size()[4]
        toPad = [diffZ // 2, diffZ - diffZ // 2, diffY // 2, diffY - diffY // 2,diffX // 2, diffX - diffX // 2]
        x = torch.nn.functional.pad(x,toPad,mode = 'constant',value = 0)
        x = torch.concat([x,dConv2],1)
        x = self.upC2T(x)

        x = self.upC3U(x)
        diffX = dConv1.size()[2] - x.size()[2]
        diffY = dConv1.size()[3] - x.size()[3]
        diffZ = dConv1.size()[4] - x.size()[4]
        toPad = [diffZ // 2, diffZ - diffZ // 2, diffY // 2, diffY - diffY // 2,diffX // 2, diffX - diffX // 2]
        x = torch.nn.functional.pad(x,toPad,mode = 'constant',value = 0)
        x = torch.concat([x,dConv1],1)
        x = self.upC3T(x)

        # final convolution to get as many output volumes as there is classes
        x = self.classConv(x)

        return x     