import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import CloudDataset

class UNET(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        
        #Contracting phase
        self.conv1 = self.contract_block(in_channels,32,3,1)
        self.conv2 = self.contract_block(32,64,3,1)
        self.conv3 = self.contract_block(64,128,3,1)
        
        #Expanding phase
        self.upconv3 = self.expand_block(128,64,3,1)
        self.upconv2 = self.expand_block(64*2,32,3,1)
        self.upconv1 = self.expand_block(32*2,out_channels,3,1)
        
    def contract_block(self,in_channels,out_channels,kernel_size,padding):
        contract = nn.Sequential( nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(),
                                nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=padding),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        
        return contract
    
    def expand_block(self,in_channels,out_channels,kernel_size,padding):
        expand = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(),
                                nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=padding),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(),
                                nn.ConvTranspose2d(out_channels,out_channels,kernel_size=3,stride=2,padding=1,output_padding=1))
        
        return expand
    
    def forward(self,xb):
        #Downsampling part
        conv1 = self.conv1(xb)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        upconv3 = self.upconv3(conv3)
        
        upconv2 = self.upconv2(torch.cat([upconv3,conv2],1))
        upconv1 = self.upconv1(torch.cat([upconv2,conv1],1))
        
        return upconv1

if __name__ == "__main__":
    unet = UNET(4,2)
    base_path = Path('../input/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training')
    data = CloudDataset(base_path/'train_red',
                    base_path/'train_green',
                    base_path/'train_blue',
                    base_path/'train_nir',
                    base_path/'train_gt')
    train_ds , val_ds = torch.utils.data.random_split(data,(6000,2400))
    print(len(train_ds) , len(val_ds))

    train_dl = DataLoader(train_ds,batch_size=2,shuffle=True,pin_memory=True,num_workers=3)
    xb , yb = next(iter(train_dl))
    print(xb.shape , yb.shape)
    print(unet)
    pred = unet(xb)
    print(pred.shape)
