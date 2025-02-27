# This code is based on the code provided by the authors of the paper "Exploiting Diffusion Prior for Real-World Image Super-Resolution" 
# in the following github repository https://github.com/IceClear/StableSR
import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import pi, exp, sqrt
from tqdm import tqdm 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from utils import get_data_patches_lr
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
import os
from torchvision import transforms
import shutil
from datetime import datetime
import time

class split_aggregation_sampling:
    def __init__(self, img_lr, patch_size, stride, batch_size, magnification_factor, device, multiple_gpus=False):
        '''
        This class is used to perform a split of an image into patches (with the patchifier function)
        and also to aggregate the super-resolution (if magnification_factor >1) of the generated patches (with the aggregation_sampling function).
        '''
        assert stride <= patch_size

        self.img_lr = img_lr
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.magnification_factor = magnification_factor
        self.multiple_gpus = multiple_gpus

        self.device = device
        channels, height, width = img_lr.shape
        self.patches_lr, self.patches_sr_infos = self.patchifier(img_lr, patch_size, stride, magnification_factor)
        self.weight = self.gaussian_weights(patch_size*magnification_factor, patch_size*magnification_factor, channels)
        self.data_loader_patches_lr = self.prepare_data_loader()

    def patchifier(self, img_to_split, patch_size, stride=None, magnification_factor=1):
        '''
        This function takes an image tensor and splits it into patches of size patch_size x patch_size.
        The stride is the number of pixels to skip between patches. If stride is not specified, it is set to patch_size
        to avoid overlapping patches. 
        If you want to perform super-resolution, the magnification_factor must be > 1. The patches_lr list doesn't change
        if you choose a magnification_factor > 1 or = 1, but the patches_sr_infos list will contain the coordinates of the
        patches in the high resolution image (otherwise, if magnification_factor=1 patches_sr_infos will contain the coordinates
        of the patches in the low resolution image).
        '''
        if stride is None:
            stride = patch_size  # Default non-overlapping behavior

        channels, height, width = img_to_split.shape
        patches_lr = []
        patches_sr_infos = []

        for y in range(0, height + 1, stride):
            for x in range(0, width + 1, stride):
                if y+patch_size > height:
                    y_start = height - patch_size
                    y_end = height
                else:
                    y_start = y
                    y_end = y+patch_size
                if x+patch_size > width:
                    x_start = width - patch_size
                    x_end = width
                else:
                    x_start = x
                    x_end = x+patch_size
                if (y_start*magnification_factor, y_end*magnification_factor, x_start*magnification_factor, x_end*magnification_factor) not in patches_sr_infos:
                    patch = img_to_split[:,  y_start:y_end, x_start:x_end]
                    patches_lr.append(patch)
                    patches_sr_infos.append((y_start*magnification_factor, y_end*magnification_factor, x_start*magnification_factor, x_end*magnification_factor))

        return patches_lr, patches_sr_infos

    def aggregation_sampling(self, model, model_name):
        '''
        This function iterates over the patches in self.patches_lr and for each patch it generates a super-resolution
        patch using the specified model. Afterwords it takes the product between the super-resolution patch and the gaussian weight
        (self.weight) and sums it to the portion of the final image (i.e. im_res) that corresponds to the patch (the position information
        is stored in self.patches_sr_infos). The same indexing used to sum the weighted super-resolution patches is used to sum the weights
        but in another tensor called pixel_count. 
        When the iteration is finished, the final image is obtained by dividing the sum of the weighted super-resolution (im_res) by the pixel_count.
        '''
        img_lr = self.img_lr
        magnification_factor = self.magnification_factor
        self.model = model
        self.model.eval()

        channels, height, width = img_lr.shape
        # Initialize two tensors of the same shape (super-resolution image shape). im_res will be the final image that will be
        # obtained by dividing the sum of the weighted super-resolution patches by the pixel_count tensor.
        im_res = torch.zeros([channels, height*magnification_factor, width*magnification_factor], dtype=img_lr.dtype, device=self.device)
        pixel_count = torch.zeros([channels, height*magnification_factor, width*magnification_factor], dtype=img_lr.dtype, device=self.device)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f'results_{model_name}_{timestamp}'

        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

        os.makedirs(folder_name, exist_ok=True)
        counter = list(np.arange(len(self.patches_lr), 0, -1)-1)
        print(f"Generating {len(self.patches_lr)} patches")

        for i, batch_patch_lr in tqdm(enumerate(self.data_loader_patches_lr), desc="Saving patches"):
            start = time.time() #################### CODE TO CHECK THE GPU TIME
            batch_patch_sr = self.model(batch_patch_lr)
            GPU_time = time.time() - start #################### CODE TO CHECK THE GPU TIME
            for patch_sr in batch_patch_sr:
                to_pil(patch_sr).save(os.path.join(folder_name, str(counter.pop())+".png"))
        
        for i in tqdm(range(len(self.patches_lr)), desc="Collage patches"):
            patch_sr = to_tensor(Image.open(os.path.join(folder_name, str(i)+".png"))).to(self.device)
            im_res[:, self.patches_sr_infos[i][0]:self.patches_sr_infos[i][1], self.patches_sr_infos[i][2]:self.patches_sr_infos[i][3]] += patch_sr * self.weight
            pixel_count[:, self.patches_sr_infos[i][0]:self.patches_sr_infos[i][1], self.patches_sr_infos[i][2]:self.patches_sr_infos[i][3]] += self.weight
        
        shutil.rmtree(folder_name)

        assert torch.all(pixel_count != 0)
        im_res /= pixel_count
        im_res = torch.clamp(im_res, 0, 1)

        return im_res, GPU_time #################### CODE TO CHECK THE GPU TIME

    def gaussian_weights(self, tile_width, tile_height, channels):
            """
            Generates a gaussian mask of weights for tile contributions
            
            This function creates two vectors (x_probs and y_probs) by taking a range from 0 to tile_width and
            a range from 0 to tile_height, and applying the gaussian function on them.
            Then the outer product is performed between y_probs and x_probs (so, a 2D matrix is created: the (0,0) element of the 2D matrix
            is the product of the 0 element of x_probs and the 0 element of y_probs).
            Finally, this matrix is tiled (the 2D matrix is repeated for the choosen shape) to have the same shape as the patches.
            """
            latent_width = tile_width
            latent_height = tile_height

            var = 0.01
            midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
            x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
            midpoint = latent_height / 2
            y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

            weights = torch.tensor(np.outer(y_probs, x_probs)).to(torch.float32).to(self.device)
            return torch.tile(weights, (channels, 1, 1))

    def prepare_data_loader(self,):
        dataset = get_data_patches_lr(self.patches_lr)
        if self.multiple_gpus:
            data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, sampler=DistributedSampler(dataset, shuffle=False))
        else:
            data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)
        return data_loader
    
def launch(args):
    from PIL import Image
    from torchvision import transforms
    from UNet_model import Residual_Attention_UNet
    import os  

    snapshot_folder_path = args.snapshot_folder_path
    model_name = args.model_name
    magnification_factor = args.magnification_factor
    patch_size = args.patch_size
    stride = args.stride
    batch_size = args.batch_size
    destination_path = args.destination_path
    img_path = args.img_path
    multiple_gpus = args.multiple_gpus
    out_dim = args.out_dim
    
    if multiple_gpus:
        print("Using multiple GPUs")
        init_process_group(backend="nccl")
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(int(device))
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")

    snapshot_path = os.path.join(snapshot_folder_path, model_name, 'snapshot.pt')
    model = Residual_Attention_UNet(image_channels=3, out_dim=out_dim, device=device).to(device)
    snapshot = torch.load(snapshot_path,map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(snapshot["MODEL_STATE"])
    model = model.to(device)
    
    if multiple_gpus:
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
    model.eval()

    img_lr = Image.open(img_path)

    transform = transforms.Compose([transforms.ToTensor()])
    img_lr = transform(img_lr).unsqueeze(0).to(device)
    aggregation_sampling = split_aggregation_sampling(img_lr, patch_size, stride, batch_size, magnification_factor, device, multiple_gpus)
    final_pred = aggregation_sampling.aggregation_sampling(model, model_name)

    if destination_path:
        final_pred = (final_pred-final_pred.min())/(final_pred.max()-final_pred.min())*255
        final_pred = final_pred.to(torch.uint8)
        final_pred = Image.fromarray(final_pred[0].permute(1,2,0).detach().cpu().numpy())
        final_pred = final_pred.filter(ImageFilter.GaussianBlur(radius=1.5))
        final_pred.save(destination_path)
    
    if multiple_gpus:
        destroy_process_group()

if __name__ == '__main__':
    import argparse 
    
    def str2bool(v):
        """Convert string to boolean."""
        return v.lower() in ("yes", "true", "t", "1")
    
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--snapshot_folder_path', type=str, default='snapshot.pt')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--magnification_factor', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument('--destination_path', type=str, default=None)
    parser.add_argument('--img_path', type=str)
    parser.add_argument("--multiple_gpus", type=str2bool, default=False, help="Use multiple GPUs")
    parser.add_argument("--out_dim", type=int, default=3, help="Output dimension of the model")
    args = parser.parse_args()
    launch(args)


