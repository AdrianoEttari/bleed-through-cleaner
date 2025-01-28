from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from skimage.filters import threshold_otsu
import cv2
import torch
# from concurrent.futures import ThreadPoolExecutor

class get_data(Dataset):
    '''
    This class allows to store the data in a Dataset that can be used in a DataLoader
    like that train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True).

    -Input:
        root_dir: path to the folder where the data is stored. 
        x_folder_name: name of the folder where the input images are stored.
        y_folder_name: name of the folder where the target images are stored.
        transform: a torchvision.transforms.Compose object with the transformations that will be applied to the images.
    -Output:
        A Dataset object that can be used in a DataLoader.

    __getitem__ returns x and y. The split in batches must be done in the DataLoader (not here).
    '''
    def __init__(self, root_dir, x_folder_name, y_folder_name, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.x_folder_name = x_folder_name
        self.y_folder_name = y_folder_name
        self.x_filenames = sorted(os.listdir(os.path.join(self.root_dir, self.x_folder_name)))
        self.y_filenames = sorted(os.listdir(os.path.join(self.root_dir, self.y_folder_name)))
    
    def __len__(self):
        return len(self.y_filenames)

    def __getitem__(self, idx):
        y_path = os.path.join(self.root_dir, self.y_folder_name, self.y_filenames[idx])
        y = np.array(Image.open(y_path))
        
        x_path = os.path.join(self.root_dir, self.x_folder_name, self.x_filenames[idx])
        x = Image.open(x_path)
        # x_HSV = np.array(x.convert('HSV'))
        # x_LAB = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2LAB)
        # x = np.concatenate((x_LAB, x_HSV[:,:,2][:,:,None]), axis=2)

        if self.transform:
            y = self.transform(y)
            x = self.transform(x)

        to_tensor = transforms.ToTensor()
        x = to_tensor(x)

        y = np.where(y==255, 1, y)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            
        return x, y
    
  
def otsu_thresholding(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding on a denoised grayscale image.
    
    Parameters:
    image (np.ndarray): Denoised grayscale image as a numpy array (2D).
    
    Returns:
    np.ndarray: Binary image after applying Otsu's thresholding.
    """
    # Ensure the input is a 2D grayscale image
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D grayscale image")
    
    # Compute Otsu's threshold
    threshold_value = threshold_otsu(image)
    
    # Apply the threshold to create a binary image
    binary_image = image > threshold_value

    # Return the binary image (foreground as 1, background as 0)
    return binary_image.astype(np.uint8)*255  # Convert to uint8 for binary image (0 and 1)

def hysteresis_thresholding(img, low_thresh, high_thresh):
    # Step 1: Apply high and low thresholds
    strong_edges = (img >= high_thresh)
    weak_edges = ((img >= low_thresh) & (img < high_thresh))

    # Create an empty output image
    result = np.zeros_like(img)

    # Mark strong edges as 255 (white)
    result[strong_edges] = 255

    # Step 2: Check connectivity of weak edges to strong edges
    def is_connected(i, j):
        for x in range(max(0, i-1), min(i+2, img.shape[0])):
            for y in range(max(0, j-1), min(j+2, img.shape[1])):
                if strong_edges[x, y]:
                    return True
        return False

    # For each weak edge, check if it is connected to a strong edge
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if weak_edges[i, j]:
                if is_connected(i, j):
                    result[i, j] = 255  # Make it a strong edge
                else:
                    result[i, j] = 0  # Suppress the weak edge

    return result

def hysteresis_thresholding_full_process(img_path, low_thresh, high_thresh):
    '''
    This function applies the hysteresis thresholding to a binary image.
    -Input:
        img: a torch tensor with the binary image.
        low_thresh: a float with the low threshold value.
        high_thresh: a float with the high threshold value.
    -Output:
        A binary image with the same size as the input image.
    '''
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.hypot(sobelx, sobely)

    # Normalize the gradient magnitude
    gradient_magnitude = gradient_magnitude / gradient_magnitude.max() * 255
    gradient_magnitude = np.uint8(gradient_magnitude)

    edges = hysteresis_thresholding(gradient_magnitude, low_thresh, high_thresh)

    return edges

def alpha_blending(img1, img2, weight1):
    return img1*weight1 + img2*(1-weight1)

def labelme_json_to_dataset(json_folder_path):
    '''
    This function allows to move the json files generated by the labelme application, to single folders with the images and the masks.
    '''
    for json_file in os.listdir(json_folder_path):
        if json_file.endswith(".json"):
            os.system("labelme_json_to_dataset "+json_folder_path+"/"+json_file)

class get_data_patches_lr(Dataset):
    def __init__(self, patches):
        self.patches = patches
    
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]
    
def weighted_total_error_function(gt_img, pred_img):
    '''
    0: pixel is foreground
    1: pixel is background or bleed-trough
    '''
    assert len(gt_img.shape) == 2
    assert len(pred_img.shape) == 2
    assert len(np.unique(gt_img)) == 2
    assert len(np.unique(pred_img)) == 2
    assert np.all([np.max(pred_img) == 1 and np.max(gt_img)==1 and np.min(gt_img)==0 and np.min(pred_img)==0])
    
    n_pixels = gt_img.size
    n_background_pixels = np.count_nonzero(gt_img)
    n_foreground_pixels = n_pixels - n_background_pixels
    assert n_foreground_pixels > 0
    assert n_background_pixels > 0
    # Foreground Error: the pixel is foreground but is classified as background or bleed-trough
    foreground_error = 0
    indexes_0 = np.argwhere(gt_img==0)
    for i,j in indexes_0:
        if pred_img[i,j]==1:
            foreground_error+=1
    foreground_error_ratio = foreground_error/n_foreground_pixels
    # Background Error: the pixel is background or bleed-through but is classified as foreground
    background_error = 0
    indexes_1 = np.argwhere(gt_img==1)
    for i,j in indexes_1:
        if pred_img[i,j]==0:
            background_error+=1
    background_error_ratio = background_error/n_background_pixels

    weighted_total_error = (n_background_pixels*background_error_ratio + n_foreground_pixels*foreground_error_ratio)/n_pixels
    
    return foreground_error_ratio, background_error_ratio, weighted_total_error

# from utils import weighted_total_error_function
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# # predicted = np.array(Image.open(r'cleaned_images_inpainting\5_BLEED_THROUGH_MASK.png'))/255
# # real = np.array(Image.open(r'DIBCO_DATA\DIBCO2018_GT\5_gt.bmp'))[:,:,0]/255
# predicted = np.array(Image.open(r'cleaned_images_inpainting\H10_BLEED_THROUGH_MASK.png'))/255
# try:
#     real = np.array(Image.open(r'DIBCO2014_GT\H10_estGT.tiff'))[:,:,0]/255
# except:
#     real = np.array(Image.open(r'DIBCO2014_GT\H10_estGT.tiff'))/255
# weighted_total_error_function(real, predicted)


def unsharp_masking(img_path, save_path):
    '''
    This function applies the unsharp masking technique to an image.
    -Input:
        img_path: a string with the path to the image.
        save_path: a string with the path to save the sharpened image.
    -Output:
        None
    '''
    image = cv2.imread(img_path)

    # Create an unsharp mask
    # STEP 1: Apply a Gaussian blur to the original image to create a softened version.
    gaussian_blur = cv2.GaussianBlur(image, (0, 0), 3)
    # STEP 2: Subtract the blurred image from the original to isolate the high-frequency details (edges).
    # The extracted details are amplified and added back to the original image, making edges sharper
    sharpened = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0) # 1.5 (original image weight), -0.5 (blurred image weight). The subtraction removes the blurred part. 0 is the scaler bias (no additional brightness adjustment).

    # Save the sharpened image
    cv2.imwrite(save_path, sharpened)
