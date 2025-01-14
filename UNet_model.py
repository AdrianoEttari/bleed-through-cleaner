import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import warnings
# from multihead_attention.MultiHeadAttention import MultiHeadAttention
# from einops import rearrange
# from multihead_attention.Visual_MultiHeadAttention import Visual_MultiHeadAttention

#########################################################################################################
#################################### Classes for all the UNet models ####################################
#########################################################################################################
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta    # set the beta parameter for the exponential moving average
        self.step = 0       # step counter (initialized at 0) to track when to start updating the moving average

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()): #iterate over all parameters in the current and moving average models
            # get the old and new weights for the current and moving average models
            old_weight, up_weight = ma_params.data, current_params.data
            # update the moving average model parameter
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        # if there is no old weight, return the new weight
        if old is None:
            return new
        # compute the weighted average of the old and new weights using the beta parameter
        return old * self.beta + (1 - self.beta) * new # beta is usually around 0.99
        # therefore the new weights influence the ma parameters only a little bit
        # (which prevents outliers to have a big effect) whereas the old weights
        # are more important.

    def step_ema(self, ema_model, model, step_start_ema=2000):
        '''
        We'll let the EMA update start just after a certain number of iterations
        (step_start_ema) to give the main model a quick warmup. During the warmup
        we'll just reset the EMA parameters to the main model one.
        After the warmup we'll then always update the weights by iterating over all
        parameters and apply the update_average function.
        '''
        # if we are still in the warmup phase, reset the moving average model to the current model
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        # otherwise update the moving average model parameters using the current model parameters
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        # reset the parameters of the moving average model to the current model parameters
        ema_model.load_state_dict(model.state_dict()) # we set the weights of ema_model
        # to the ones of model.

class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_x, f_int, device):
        '''
        AttentionBlock: Applies an attention mechanism to the input data.
        
        Args:
            f_g (int): Number of channels in the 'g' input (image on the up path).
            f_x (int): Number of channels in the 'x' input (residual image).
            f_int (int): Number of channels in the intermediate layer.
            device: Device where the operations should be performed.
        '''
        super().__init__()
        self.device = device
        self.w_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True).to(device),
        ) # Computes a 1x1 convolution of the 'g' input to reduce its channel dimension to f_int.
        
        self.w_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, kernel_size=2, stride=2, padding=0, bias=True).to(device),
        ) # Computes a 1x1 convolution of the 'x' input to reduce its channel dimension to f_int.

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True).to(device),
            nn.Sigmoid()
        ) # Computes a 1x1 convolution of the element-wise sum of the processed 'g' and 'x' inputs, followed by a sigmoid activation.
        
        self.relu = nn.ReLU(inplace=False)

        self.result = nn.Sequential(
            nn.Conv2d(f_x, f_x, kernel_size=1, stride=1, padding=0, bias=True).to(device),
            nn.BatchNorm2d(f_x).to(device)
        )
                                                                        
    def forward(self, x, g):
        '''
        Forward pass for the AttentionBlock.

        Args:
            x (torch.Tensor): The 'x' input (residual image).
            g (torch.Tensor): The 'g' input (image on the up path).

        Returns:
            torch.Tensor: The output of the attention mechanism applied to the input data.
        '''
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        if g1.shape != x1.shape:
            warnings.warn("The shapes of g1 and x1 are different.")
            x1 = resize(x1.to('cpu'), size=(g1.shape[2], g1.shape[3])).to(self.device)
        psi = self.relu(g1 + x1) 
        psi = self.psi(psi) 
        upsample_psi = F.interpolate(psi, scale_factor=2, mode='nearest') 
        upsample_psi = upsample_psi.repeat_interleave(repeats=x.shape[1], dim=1) 
        if upsample_psi.shape != x.shape:
            warnings.warn("The shapes of upsample_psi and x are different.")
            upsample_psi = resize(upsample_psi.to('cpu'), size=(x.shape[2], x.shape[3])).to(self.device)
        result = self.result(upsample_psi * x) 

        return result
     
class ResConvBlock(nn.Module):
    '''
    This class defines a residual convolutional block. It does not contain the layer for the actual
    downsampling: it doesn't contain a layer which shrinks the spatial dimensions of the input data.
    '''
    def __init__(self, in_ch, out_ch, device):
        super().__init__()
        self.batch_norm1 = nn.BatchNorm2d(out_ch, device=device)
        self.batch_norm2 = nn.BatchNorm2d(out_ch, device=device)
        self.shortcut_batch_norm = nn.BatchNorm2d(out_ch, device=device)
        self.relu = nn.ReLU(inplace=False) # inplace=True MEANS THAT IT WILL MODIFY THE INPUT DIRECTLY, WITHOUT ASSIGNING IT TO A NEW VARIABLE (THIS SAVES SPACE IN MEMORY, BUT IT MODIFIES THE INPUT)
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_ch, out_ch,
                                            kernel_size=3, stride=1,
                                            padding='same', bias=True,
                                            device=device),
                                  self.batch_norm1,
                                  self.relu)
        self.conv_skip = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Sequential(
                                  nn.Conv2d(out_ch, out_ch,
                                            kernel_size=3, stride=1,
                                            padding='same', bias=True,
                                            device=device),
                                  self.batch_norm2)
        self.shortcut_conv = nn.Sequential(
                                    nn.Conv2d(in_ch, out_ch, 
                                        kernel_size=1, stride=1,
                                        padding='same', bias=True,
                                        device=device),
                                    self.shortcut_batch_norm)
    
    def forward(self, x, x_skip):
        # FIRST CONV
        h = self.conv1(x)
        # SUM THE X-SKIP IMAGE WITH THE INPUT IMAGE
        if x_skip is not None:
            x_skip = self.conv_skip(x_skip)
            h = h + x_skip
        # SECOND CONV
        h = self.conv2(h)
        # SHORTCUT
        shortcut = self.shortcut_conv(x)
        # SUM THE SHORTCUT WITH THE OUTPUT OF THE SECOND CONV AND APPLY ACTIVATION FUNCTION
        output = self.relu(shortcut + h)
        return output
    
class UpConvBlock(nn.Module):
    '''
    This class performs a convolution and a transposed convolution on the input data. The latter
    increases the spatial dimensions of the data by a factor of 2.
    '''
    def __init__(self, in_ch, out_ch, device):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(out_ch, device=device)
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True, device=device)
        self.transform = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True, output_padding=1, device=device)

    
    def forward(self, x):
        x = self.relu(self.batch_norm(self.conv(x)))
        output = self.transform(x)
        return output
    
class gating_signal(nn.Module):
    '''
    This class is used to generate a gating signal that is used in the attention mechanism.
    It just applies a 1x1 convolution followed by a batch normalization and a ReLU activation that 
    moves the depth dimension of the input tensor from in_dim to out_dim.
    '''
    def __init__(self, in_dim, out_dim, device):
        super(gating_signal, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding='same', device=device)
        self.batch_norm = nn.BatchNorm2d(out_dim, device=device)
        self.relu = nn.ReLU(inplace=False)
        self.device = device

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.relu(x)

#########################################################################################################
################################################ Models #################################################
#########################################################################################################
class Residual_Attention_UNet(nn.Module):
    def __init__(self, image_channels=3, out_dim=3, device=None):
        super().__init__()
        self.image_channels = image_channels
        self.down_channels = (16,32,64,128,256) # Note that there are 4 downsampling layers and 4 upsampling layers.
        # To understand why len(self.down_channels)=5, you have to imagine that the first layer 
        # has a Conv2D(16,32), the second layer has a Conv2D(32,64) and the third layer has a Conv2D(64,128)...
        self.up_channels = (256,128,64,32,16) # Note that the last channel is not used in the upsampling (it goes from up_channels[-2] to out_dim)
        self.out_dim = out_dim 
        self.device = device
        # It's important to note that the dimensionality of time embeddings should be chosen carefully,
        # considering the trade-off between model complexity and the amount of available data.

        # INITIAL PROJECTION
        self.conv0 = nn.Conv2d(self.image_channels, self.down_channels[0], 3, padding=1) # SINCE THERE IS PADDING 1 AND STRIDE 1,  THE OUTPUT IS THE SAME SIZE OF THE INPUT
        
        # DOWNSAMPLE
        self.conv_blocks = nn.ModuleList([
            ResConvBlock(in_ch=self.down_channels[i],
                      out_ch=self.down_channels[i+1],
                      device=self.device) \
            for i in range(len(self.down_channels)-2)])
        
        self.downs = nn.ModuleList([
            nn.Conv2d(self.down_channels[i+1], self.down_channels[i+1], kernel_size=3, stride=2, padding=1, bias=True, device=device)\
        for i in range(len(self.down_channels)-2)])

        # BOTTLENECK
        self.bottle_neck = ResConvBlock(in_ch=self.down_channels[-2],
                                        out_ch=self.down_channels[-1],
                                        device=self.device)
        
        # UPSAMPLE
        self.gating_signals = nn.ModuleList([
            gating_signal(self.up_channels[i], self.up_channels[i+1], self.device) \
            for i in range(len(self.up_channels)-2)])
        
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(self.up_channels[i+1], self.up_channels[i+1], self.up_channels[i+1], self.device) \
            for i in range(len(self.up_channels)-2)])
        
        self.ups = nn.ModuleList([
            UpConvBlock(in_ch=self.up_channels[i], out_ch=self.up_channels[i], device=self.device) \
            for i in range(len(self.up_channels)-2)])
        
        self.up_convs = nn.ModuleList([
            nn.Conv2d(int(self.up_channels[i]*3/2), self.up_channels[i+1], kernel_size=3, stride=1, padding=1, bias=True).to(self.device) \
            for i in range(len(self.up_channels)-2)])

        # OUTPUT
        self.output_activation = nn.Sigmoid()
        self.output = nn.Conv2d(self.up_channels[-2], self.out_dim, 1)
    
    def forward(self, x):

        # INITIAL CONVOLUTION
        x = self.conv0(x)
        
        # SKIP CONNECTION
        x_skip = x.clone()

        # UNET (DOWNSAMPLE)        
        residual_inputs = []
        for i, (conv_block, down) in enumerate(zip(self.conv_blocks, self.downs)):
            if i == 0:
                x = conv_block(x, x_skip)
            else:
                x = conv_block(x, None)
            residual_inputs.append(x)
            x = down(x)
        
        # UNET (BOTTLENECK)
        x = self.bottle_neck(x, None)

        # UNET (UPSAMPLE)
        for i, (gating_signal, attention_block, up, up_conv) in enumerate(zip(self.gating_signals,self.attention_blocks,self.ups, self.up_convs)):
            gating = gating_signal(x)
            attention = attention_block(residual_inputs[-(i+1)], gating)
            x = up(x)
            if attention.shape[-1] != x.shape[-1] or attention.shape[-2] != x.shape[-2]:
                warnings.warn("The shapes of attention and x are different.")
                x = resize(x.to('cpu'), size=(attention.shape[2], attention.shape[3])).to(self.device)
            x = torch.cat([x, attention], dim=1)
            x = up_conv(x)

        return self.output_activation(self.output(x))
        # return self.output(x)
 


if __name__=="__main__":
    device='cuda'
    model = Residual_Attention_UNet(image_channels=3, out_dim=3, device=device).to(device)
    print("Num params: ", sum(p.numel() for p in model.parameters()))



