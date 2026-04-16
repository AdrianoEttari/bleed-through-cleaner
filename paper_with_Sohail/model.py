#%%
import torch
import torch.nn as nn

class GatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.feature = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        return self.feature(x) * torch.sigmoid(self.gate(x))
    
class InpaintUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = GatedConv2d(5, 64, 3, padding=1)
        self.enc2 = GatedConv2d(64, 128, 3, stride=2, padding=1)
        self.enc3 = GatedConv2d(128, 256, 3, stride=2, padding=1)

        self.bottleneck = GatedConv2d(256, 256, 3, padding=1)

        self.dec3 = GatedConv2d(256+256, 128, 3, padding=1)
        self.dec2 = GatedConv2d(128+128, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64+64, 3, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.dec3(torch.cat([b, e3], 1))
        d3 = F.interpolate(d3, scale_factor=2)

        d2 = self.dec2(torch.cat([d3, e2], 1))
        d2 = F.interpolate(d2, scale_factor=2)

        out = self.dec1(torch.cat([d2, e1], 1))
        return out
