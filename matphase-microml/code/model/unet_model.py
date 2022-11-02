
"""***** Model code adapted from ******

https://github.com/milesial/Pytorch-UNet/tree/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db

Full assembly of the parts to form the complete network 
"""
import torch.nn.functional as F
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, drop_train=True, drop_rate=0.5, bilinear=True):
        super(UNet, self).__init__()
        #super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.drop_train = drop_train
        self.drop_rate = drop_rate
        
        #adding 1X1 conv to input representation
        self.softconv = nn.Conv2d(in_channels=n_channels, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.inc = DoubleConv(32, 64)
        #self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.drop1 = DropOut(drop_rate,drop_train)
        self.down2 = Down(128, 256)
        self.drop2 = DropOut(drop_rate,drop_train)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.drop3 = DropOut(drop_rate,drop_train)
        self.down4 = Down(512, 1024 // factor)
        self.drop4 = DropOut(drop_rate,drop_train)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.drop5 = DropOut(drop_rate,drop_train)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, input_data):
        #F.dropout(x2,p=self.dropout,training=self.drop_train)
        x = self.softconv(input_data)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x22 = self.drop1(x2)
        x3 = self.down2(x22)
        x33 = self.drop2(x3)
        x4 = self.down3(x33)
        x44 = self.drop3(x4)
        x5 = self.down4(x44)
        x55= self.drop4(x5)
        x = self.up1(x55, x4)
        x = self.drop5(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
