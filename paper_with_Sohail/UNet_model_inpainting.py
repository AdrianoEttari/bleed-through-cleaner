import torch.nn as nn
import torch


#########################################################################################################
#################################### Classes for all the UNet models ####################################
#########################################################################################################

class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, device="cuda"):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, device=device),
            # nn.BatchNorm2d(out_ch, device=device),
            nn.InstanceNorm2d(out_ch, device=device),
            self.relu
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, device=device),
            # nn.BatchNorm2d(out_ch, device=device)
            nn.InstanceNorm2d(out_ch, device=device)
        )

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, device=device),
                # nn.BatchNorm2d(out_ch, device=device)
                nn.InstanceNorm2d(out_ch, device=device),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return self.relu(h + self.shortcut(x))

    
class gating_signal(nn.Module):
    '''
    This class is used to generate a gating signal that is used in the attention mechanism.
    It just applies a 1x1 convolution followed by a batch normalization and a ReLU activation that 
    moves the depth dimension of the input tensor from in_dim to out_dim.
    '''
    def __init__(self, in_dim, out_dim, device):
        super(gating_signal, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding='same', device=device)
        # self.batch_norm = nn.BatchNorm2d(out_dim, device=device)
        self.InstanceNorm = nn.InstanceNorm2d(out_dim, device=device)
        self.relu = nn.ReLU(inplace=False)
        self.device = device

    def forward(self, x):
        x = self.conv(x)
        # x = self.batch_norm(x)
        x = self.InstanceNorm(x)
        return self.relu(x)

#########################################################################################################
################################################ Models #################################################
#########################################################################################################



class ResidualUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,
                 channels=(32, 64, 128, 256), device="cuda"):
        super().__init__()

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, channels[0], 3, padding=1, device=device)

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.enc_blocks.append(
                ResConvBlock(channels[i], channels[i + 1], device=device)
            )
            self.downs.append(
                nn.Conv2d(
                    channels[i + 1],
                    channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    device=device
                )
            )

        # Bottleneck
        self.bottleneck = ResConvBlock(channels[-1], channels[-1], device=device)

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for enc_ch, dec_ch in zip(reversed(channels[:-1]),
                                  reversed(channels[1:])):
            self.ups.append(
                nn.ConvTranspose2d(dec_ch, enc_ch, kernel_size=2, stride=2, device=device)
            )
            self.dec_blocks.append(
                ResConvBlock(enc_ch * 2, enc_ch, device=device)
            )
        
        self.gatings = nn.ModuleList()
        for i in range(len(channels)-1):
            self.gatings.append(
                gating_signal(channels[::-1][i], channels[::-1][i+1], device=device)
            )
        # Output
        self.output = nn.Conv2d(channels[0], out_channels, 1, device=device)

    def forward(self, x):
        x = self.conv0(x)

        skips = []
        for block, down in zip(self.enc_blocks, self.downs):
            x = block(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)
        
        # print("Shape of x after bottleneck: ", x.shape)
        # for i, skip in enumerate(skips):
        #     print(f"Shape of skip connection {i}: ", skip.shape)

        for i, (up, gating, block, skip) in enumerate(zip(self.ups, self.gatings, self.dec_blocks, reversed(skips))):
            x = up(x)
            skip = gating(skip)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.output(x)

 


if __name__=="__main__":
    pass



