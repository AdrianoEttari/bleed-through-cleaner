from PIL import Image, ImageFilter
import numpy as np
from UNet_model import Residual_Attention_UNet
import os
import torch
from torchvision import transforms
from utils import otsu_thresholding
import matplotlib.pyplot as plt
from Aggregation_Sampling import split_aggregation_sampling
import cv2
from tqdm import tqdm
from astropy.stats import biweight_location, sigma_clip
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter
import warnings
import shutil

class bleed_through_cleaner:
    def __init__(self, image_path, models_folder_path, device) -> None:
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.models_folder_path = models_folder_path
        self.device = device

    def page_extract(self, model_name):
        '''
        In this function a rescaled version of the input image is given to the page extraction model that returns a segmentation mask of the page.
        The mask is then passed to an otsu thresholding function which returns a binary mask of the page. 
        The binary mask is adjusted with morphological operations and then is passed to the page_filter function that returns the coordinates of a
        polygon that contains the page, excluding what's outside the page. 
        The coordinates are also scaled to the original scale and the same for the binary mask.
        Eventually, the original image and the binary mask are cropped according to the scaled coordinates.
        '''
        target_pixels = 6*10**5
        number_of_pixels = self.image.size[0]*self.image.size[1]
        scale = np.sqrt(target_pixels/number_of_pixels)
        input_image_rescaled = self.image.resize((int(self.image.size[0]*scale), int(self.image.size[1]*scale)))
        snapshot_path = os.path.join(self.models_folder_path, model_name, 'snapshot.pt')
        model_page_extractor = Residual_Attention_UNet(image_channels=3, out_dim=1, device=self.device).to(self.device)
        assert os.path.exists(snapshot_path), f"Snapshot not found at {snapshot_path}"
        snapshot = torch.load(snapshot_path, map_location=torch.device('cpu'), weights_only=True)
        model_page_extractor.load_state_dict(snapshot["MODEL_STATE"])
        model_page_extractor = model_page_extractor.to(self.device)
        model_page_extractor.eval()
        transform = transforms.Compose([transforms.ToTensor(),])
        input_image_rescaled_tensor = transform(input_image_rescaled).unsqueeze(0).to(self.device)
        mask_page = model_page_extractor(input_image_rescaled_tensor)[0].permute(1,2,0).squeeze(2).detach().cpu().numpy()

        otsu_mask_page = otsu_thresholding(mask_page) 

        # cv2.morphologyEx with cv2.MORPH_OPEN performs opening (erosion followed by dilation)
        # which is useful to remove noise.
        kernel = np.ones((5,5),np.uint8)
        otsu_mask_cleaned = cv2.morphologyEx(otsu_mask_page, cv2.MORPH_OPEN, kernel)
        # cv2.connectedComponentsWithStats is used in this case to get stats where there are also the
        # number of pixels for each label found in the image (where labels are the connected components
        # and so if there is a page and a single little agglomerate of pixels not linked to the page, then
        # there will be 3 classes because we must also consider the background). Once we have the number of 
        # pixels for each label, we can just take the label that is not the background with the most pixels.
        # Finally we can filter the mask with the largest component.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu_mask_cleaned)
        largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        otsu_mask_cleaned = (labels == largest_component).astype("uint8")

        x_left, x_right, y_up, y_down = self.page_filter(otsu_mask_cleaned)
        x_left = int(x_left/scale)
        x_right = int(x_right/scale)
        y_up = int(y_up/scale)
        y_down = int(y_down/scale)
        otsu_mask_cleaned = Image.fromarray(otsu_mask_cleaned).resize((self.image.size[0], self.image.size[1]))
        otsu_mask_cleaned = np.array(otsu_mask_cleaned)[y_up:y_down, x_left:x_right]
        page_filtered_image = np.array(self.image)[y_up:y_down, x_left:x_right]
        return page_filtered_image, otsu_mask_cleaned

    def page_filter(self, mask_page):
        '''
        This function takes a binary mask and returns the coordinates of the polygon that contains the page excluding 
        the outside of the page (e.g. the table where the book is placed).

        How it works:
            * it moves from left to right, column by column and sums the values of the pixels in the column. If the sum is
            0, it means that the column is empty and so it increments a counter. If the sum is not 0, it means that the column
            contains a portion of the page and so we must stop the iterator because we have found the leftmost column of the page
            whose index is indicated by the counter.

            * it moves from right to left, column by column and sums the values of the pixels in the column. If the sum is
            0, it means that the column is empty and so it increments a counter. If the sum is not 0, it means that the column
            contains a portion of the page and so we must stop the iterator because we have found the rightmost column of the page
            whose index is indicated by the counter.

            * it moves from up to down, row by row and sums the values of the pixels in the row. If the sum is 0, it means that
            the row is empty and so it increments a counter. If the sum is not 0, it means that the row contains a portion of the page
            and so we must stop the iterator because we have found the uppermost row of the page whose index is indicated by the counter.

            * it moves from down to up, row by row and sums the values of the pixels in the row. If the sum is 0, it means that
            the row is empty and so it increments a counter. If the sum is not 0, it means that the row contains a portion of the page
            and so we must stop the iterator because we have found the lowermost row of the page whose index is indicated by the counter.
        '''
        counter_x_left = 0
        for col in range(mask_page.shape[1]):
            if np.sum(mask_page[:,col])==0:
                counter_x_left += 1
            else:
                break

        counter_x_right = 0
        for col in range(mask_page.shape[1])[::-1]:
            if np.sum(mask_page[:,col])==0:
                counter_x_right += 1
            else:
                break
        
        counter_y_up = 0
        for raw in range(mask_page.shape[0]):
            if np.sum(mask_page[raw,:])==0:
                counter_y_up += 1
            else:
                break
        
        counter_y_down = 0
        for raw in range(mask_page.shape[0])[::-1]:
            if np.sum(mask_page[raw,:])==0:
                counter_y_down += 1
            else:
                break
        
        x_left = counter_x_left
        x_right = mask_page.shape[1]-counter_x_right
        y_up = counter_y_up
        y_down = mask_page.shape[0]-counter_y_down

        return x_left, x_right, y_up, y_down

    def ornament_detect(self, aggregation_sampling, model_name):
        '''
        This function build an ornament segmentation mask using the numpy_img given as input. The model used has the name model_name.
        Structure of the function:
            * Load the model and the snapshot. Load the snapshots weights into the model and set the model in eval mode.
            * Load the split_aggregation_sampling class (aggregation_sampling argument) and use the aggregation_sampling function of this class to perform
            the inference on the patches and aggregate the results.
            * The final prediction is then post-processed by normalizing the values between 0 and 255 and then converting them to uint8. Moreover,
            some gaussian blur is applied in order to smooth the prediction that sometimes can present some noise. The result has 3 channels but they
            are the same, so we take just the first of them.
            * Up to now we have a mask not binary. To build a binary mask we perfom thresholding. The ornament mask is thresholded using a simple threshold
            (the values can go from 0 to 255 and so we can use a threshold of 0.6*255).
            * Some more post-processing is performed using morphological operations to clean the mask. Specifically we use opening that is erosion
            followed by dilation and closing that is dilation followed by erosion.
            * The function return 255-threshold_ornament_mask_cleaned because the mask must have 0 value for the ornaments and 255 for the background.
        '''
        
        snapshot_path = os.path.join(self.models_folder_path, model_name, 'snapshot.pt')
        model = Residual_Attention_UNet(image_channels=3, out_dim=1, device=self.device).to(self.device)
        snapshot = torch.load(snapshot_path,map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(snapshot["MODEL_STATE"])
        model = model.to(self.device)
        model.eval()

        final_pred = aggregation_sampling.aggregation_sampling(model, model_name)
        # final_pred = (final_pred-final_pred.min())/(final_pred.max()-final_pred.min())*255 
        # final_pred = final_pred.to(torch.uint8)
        # final_pred = Image.fromarray(final_pred[0].permute(1,2,0).detach().cpu().numpy())
        # final_pred = final_pred.filter(ImageFilter.GaussianBlur(radius=1.5))

        ornament_mask = final_pred[0].permute(1,2,0).detach().cpu().numpy()[:,:,0] # it has 3 channels but they are the same

        # For the ornament mask is better to use the classic thresholding instead of using the otsu thresholding (empirical motivation).
        threshold_ornament_mask = ornament_mask > 0.6  
        threshold_ornament_mask = threshold_ornament_mask.astype(np.uint8)  
        kernel = np.ones((15,15),np.uint8)
        threshold_ornament_mask_cleaned = cv2.morphologyEx(threshold_ornament_mask, cv2.MORPH_OPEN, kernel)
        threshold_ornament_mask_cleaned = cv2.morphologyEx(threshold_ornament_mask_cleaned, cv2.MORPH_CLOSE, kernel)

        return 255-threshold_ornament_mask_cleaned*255 # The mask has 0 value for the ornaments and 255 for the background 

    def text_detect(self, aggregation_sampling, model_name):
        '''
        This function builds a text segmentation mask using the numpy_img given as input. The model used has the name model_name.
        Structure of the function:
            * Load the model and the snapshot. Load the snapshots weights into the model and set the model in eval mode.
            * Load the split_aggregation_sampling class (aggregation_sampling argument) and use the aggregation_sampling function of this class to perform
            the inference on the patches and aggregate the results.
            * The final prediction is then post-processed by normalizing the values between 0 and 255 and then converting them to uint8. Moreover,
            some gaussian blur is applied in order to smooth the prediction that sometimes can present some noise. The result has 3 channels but they
            are the same, so we take just the first of them.
            * Up to now we have a mask not binary. To build a binary mask we perfom thresholding. The text mask is thresholded using a simple threshold
            (the values can go from 0 to 255 and so we can use a threshold of 0.6*255).
            * Some more post-processing is performed using morphological operations to clean the mask. Specifically we use opening that is erosion
            followed by dilation and closing that is dilation followed by erosion.
        '''
        input_channels = aggregation_sampling.patches_lr[0].shape[1]
        snapshot_path = os.path.join(self.models_folder_path, model_name, 'snapshot.pt')
        model = Residual_Attention_UNet(image_channels=input_channels, out_dim=1, device=self.device).to(self.device)
        snapshot = torch.load(snapshot_path,map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(snapshot["MODEL_STATE"])
        model = model.to(self.device)
        model.eval()

        final_pred = aggregation_sampling.aggregation_sampling(model, model_name)

        # final_pred = (final_pred-final_pred.min())/(final_pred.max()-final_pred.min())*255 ########## TO BE MODIFIED ##########

        # final_pred = final_pred[0].permute(1,2,0).detach().cpu().numpy()[:,:,0]*255
        # final_pred = Image.fromarray(final_pred.astype(np.uint8))
        # final_pred = final_pred.filter(ImageFilter.GaussianBlur(radius=1.5))
        # text_mask = np.array(final_pred)

        text_mask = final_pred[0].permute(1,2,0).detach().cpu().numpy()[:,:,0]*255 # it has 3 channels but they are the same
        text_mask = text_mask.astype(np.uint8)
   
        # thresholded_text_mask = cv2.adaptiveThreshold(text_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                   cv2.THRESH_BINARY, blockSize=21, C=4) 

        ret,thresholded_text_mask = cv2.threshold(text_mask,50,255,cv2.THRESH_BINARY)

        return thresholded_text_mask

    def bleed_through_finder(self,
                                page_extraction_model_name='Residual_attention_UNet_page_extraction',
                                ornament_model_name='Residual_attention_UNet_ornament_extraction',
                                text_model_name='Residual_attention_UNet_text_extraction'):
        '''
        This function builds the final bleed-through mask using the page_extraction_model_name, the ornament_model_name and the text_model_name.
        The page_extraction_model_name returns the filtered page and the mask of the page (that won't be used). 
        The ornament_model_name returns the ornament mask and the text_model_name returns the text mask.
        In this function it is considered also the case in which the page_filtered_image uses input image in a different color space (HSV or LAB). 
        The final bleed-through mask is built by summing the ornament mask and the text mask, then subtracting 255 to the sum and clipping the values
        in the range 0-255.
        The reason for the subtraction and the clipping is the following:
            If one mask has 0 value for a pixel (so it is an important one) and the other mask has value 255 (not important pixel), then their sum is 255
            and we want to have 0 for the bleed-through mask because it is an important one. If both have 0 then it is important for both and the final
            mask must be 0. If both have 255 then it is not important for both and the final mask must be 255. Following this ratio, we can subtract 255
            to the sum of the masks and then clip the values to be between 0 and 255 in order to get what we want.
        The function returns the page_filtered_image and the final bleed-through mask.
        '''
        patch_size = 400
        stride = 100
        if page_extraction_model_name:
            page_filtered_image, mask_page = self.page_extract(page_extraction_model_name)
        
            if page_filtered_image.shape[0] < patch_size or page_filtered_image.shape[1] < patch_size:
                raise ValueError('The page filtered image is smaller than 400 pixels in one of the two dimensions. You can deactivate the page filtering (by writing None in the page_extraction_model_name argument) if the original image is bigger than 400 in both dimensions.')
        
        else:
            page_filtered_image = np.array(self.image)
        
        transform = transforms.Compose([transforms.ToTensor()])
        page_filtered_image_tensor = transform(page_filtered_image).unsqueeze(0).to(self.device)

        if page_filtered_image_tensor.shape[2] < patch_size or page_filtered_image_tensor.shape[3] < patch_size:
            target = min([page_filtered_image_tensor.shape[2], page_filtered_image_tensor.shape[3]])
            possible_patch_size = np.arange(50,target,50)
            differences = []
            for i in possible_patch_size:
                differences.append(target-i)
            patch_size = possible_patch_size[np.argmin(differences)]
            stride = patch_size // 4
            warnings.warn(f"The patch size for this picture has been changed from 400 to {patch_size} because of the shape of the image")
            
        aggregation_sampling = split_aggregation_sampling(img_lr=page_filtered_image_tensor, patch_size=patch_size, stride=stride, magnification_factor=1, device=self.device, multiple_gpus=False)
        ornament_mask = self.ornament_detect(aggregation_sampling, ornament_model_name)

        if 'HSV' in text_model_name:
            page_filtered_image = Image.fromarray(page_filtered_image).convert('HSV')
            page_filtered_image = np.array(page_filtered_image)
        elif 'LAB' in text_model_name:
            x_HSV = np.array(Image.fromarray(page_filtered_image).convert('HSV'))
            x_LAB = cv2.cvtColor(np.array(Image.fromarray(page_filtered_image)), cv2.COLOR_RGB2LAB)
            page_filtered_image = np.concatenate((x_LAB, x_HSV[:,:,2][:,:,None]), axis=2)

        text_mask = self.text_detect(aggregation_sampling, text_model_name)

        FINAL_BLEED_THROUGH_MASK = np.zeros((page_filtered_image.shape[0], page_filtered_image.shape[1]))
        FINAL_BLEED_THROUGH_MASK += ornament_mask
        FINAL_BLEED_THROUGH_MASK += text_mask
        FINAL_BLEED_THROUGH_MASK = np.clip(FINAL_BLEED_THROUGH_MASK-255, 0, 255).astype(np.uint8)

        if 'HSV' in text_model_name:
            page_filtered_image_RGB = cv2.cvtColor(page_filtered_image, cv2.COLOR_HSV2RGB)
        elif 'LAB' in text_model_name:
            x_HSV = page_filtered_image[:,:,-1]
            x_LAB = page_filtered_image[:,:,:3]
            page_filtered_image_RGB = cv2.cvtColor(x_LAB, cv2.COLOR_LAB2RGB)
        
        return page_filtered_image, FINAL_BLEED_THROUGH_MASK

    def median_image_inpainting(self,
                                page_extraction_model_name='Residual_attention_UNet_page_extraction',
                                ornament_model_name='Residual_attention_UNet_ornament_extraction',
                                text_model_name='Residual_attention_UNet_text_extraction',
                                save_folder_path_mask_page = None):
        '''
        This function performs the inpainting by simply replacing the pixels in the page_filtered_image that are white in the 
        bleed-through mask with the median of the pixels in the page_filtered_image. This is done channel by channel.
        '''

        if save_folder_path_mask_page:
            page_filtered_save_path = os.path.join(save_folder_path_mask_page, os.path.basename(self.image_path).split('.')[0]+'_page_filtered.png')
            mask_bleed_through_save_path = os.path.join(save_folder_path_mask_page, os.path.basename(self.image_path).split('.')[0]+'_BLEED_THROUGH_MASK.png')
            # page_filtered_save_path = os.path.join(save_folder_path_mask_page, os.path.dirname(self.image_path).split('_')[0]+'_'+os.path.basename(self.image_path).split('_')[1]+'_page_filtered.png')
            # mask_bleed_through_save_path = os.path.join(save_folder_path_mask_page, os.path.dirname(self.image_path).split('_')[0]+'_'+os.path.basename(self.image_path).split('_')[1]+'_BLEED_THROUGH_MASK.png')

            if os.path.exists(page_filtered_save_path) and os.path.exists(mask_bleed_through_save_path):
                page_filtered_image = np.array(Image.open(page_filtered_save_path))
                mask = np.array(Image.open(mask_bleed_through_save_path))
            else:
                page_filtered_image, mask = self.bleed_through_finder(page_extraction_model_name=page_extraction_model_name,
                                                ornament_model_name=ornament_model_name,
                                                text_model_name=text_model_name)
                Image.fromarray(page_filtered_image).save(page_filtered_save_path)
                Image.fromarray(mask).save(mask_bleed_through_save_path)
        else:
            page_filtered_image, mask = self.bleed_through_finder(page_extraction_model_name=page_extraction_model_name,
                                            ornament_model_name=ornament_model_name,
                                            text_model_name=text_model_name)

        binary_mask = Image.fromarray(mask.astype(np.uint8))

        cleaned_image = np.array(page_filtered_image).copy()

        for channel in range(3):
            cleaned_image[:, :, channel] = np.where(binary_mask, np.median(cleaned_image[:, :, channel]), cleaned_image[:, :, channel]) 

        # fig, axs = plt.subplots(1,2, figsize=(15,10))
        # axs[0].imshow(page_filtered_image, cmap='gray')
        # axs[0].set_title('Original Image')
        # axs[1].imshow(cleaned_image, cmap='gray')
        # axs[1].set_title('Inpainted Image')
        # plt.show()
        return cleaned_image
    
    def GMM_image_inpainting(self,
                                page_extraction_model_name='Residual_attention_UNet_page_extraction',
                                ornament_model_name='Residual_attention_UNet_ornament_extraction',
                                text_model_name='Residual_attention_UNet_text_extraction',
                                save_folder_path_mask_page = None):
        '''
        This function takes the bleed-through mask returned by bleed_through_finder and filters out the pixels of page_filtered_image where 
        the mask is equal to 0 (important pixels); in the remaining pixels (the one with values 1 in the mask) a Gaussian Mixture Model is used
        to separate the background pixels from the bleed-through ones.
        Afterwords, a new mask is returned and just in the areas where the bleed-through mask is 1 and where the GMM
        predicts that the pixel is bleed-through, the pixel is inpainted using the median of the pixels of the image.
        This is done channel by channel. 
        '''
 
        if save_folder_path_mask_page:
            # page_filtered_save_path = os.path.join(save_folder_path_mask_page, os.path.dirname(self.image_path).split('_')[0]+'_'+os.path.basename(self.image_path).split('_')[1]+'_page_filtered.png')
            # mask_bleed_through_save_path = os.path.join(save_folder_path_mask_page, os.path.dirname(self.image_path).split('_')[0]+'_'+os.path.basename(self.image_path).split('_')[1]+'_BLEED_THROUGH_MASK.png')
            page_filtered_save_path = os.path.join(save_folder_path_mask_page, os.path.basename(self.image_path).split('.')[0]+'_page_filtered.png')
            mask_bleed_through_save_path = os.path.join(save_folder_path_mask_page, os.path.basename(self.image_path).split('.')[0]+'_BLEED_THROUGH_MASK.png')

            if os.path.exists(page_filtered_save_path) and os.path.exists(mask_bleed_through_save_path):
                page_filtered_image = np.array(Image.open(page_filtered_save_path))
                mask = np.array(Image.open(mask_bleed_through_save_path))
            else:
                page_filtered_image, mask = self.bleed_through_finder(page_extraction_model_name=page_extraction_model_name,
                                                ornament_model_name=ornament_model_name,
                                                text_model_name=text_model_name)
                Image.fromarray(page_filtered_image).save(page_filtered_save_path)
                Image.fromarray(mask).save(mask_bleed_through_save_path)
        else:
            page_filtered_image, mask = self.bleed_through_finder(page_extraction_model_name=page_extraction_model_name,
                                            ornament_model_name=ornament_model_name,
                                            text_model_name=text_model_name)

        if 'HSV' in text_model_name:
            page_filtered_image = np.array(page_filtered_image.convert('HSV'))
        elif 'LAB' in text_model_name:
            x_HSV = np.array(page_filtered_image.convert('HSV'))
            x_LAB = cv2.cvtColor(np.array(page_filtered_image), cv2.COLOR_RGB2LAB)
            page_filtered_image = np.concatenate((x_LAB, x_HSV[:,:,2][:,:,None]), axis=2)
        else:
            page_filtered_image = np.array(page_filtered_image)

        binary_mask = Image.fromarray(mask.astype(np.uint8))
        cleaned_image = np.array(page_filtered_image).copy()
        binary_mask = np.array(binary_mask)

        num_channels = int(cleaned_image.shape[2])
        cleaned_image = cleaned_image / 255
        image_to_inpaint = cleaned_image.copy()
        image_to_inpaint[binary_mask == 0] = 0 # Zero out the important regions

        inpainted_image = cleaned_image.copy()

        def build_mask_from_probs(probs, starting_mask, array_to_copy_shape):
            threshold = 0.5  
            bleed_through_mask = (probs > threshold).astype(np.uint8)
            bleed_through_mask_2d = np.zeros_like(array_to_copy_shape)
            bleed_through_mask_2d[starting_mask == 255] = bleed_through_mask.flatten()
            bleed_through_mask_2d = bleed_through_mask_2d.astype(np.uint8)
            return bleed_through_mask_2d
    
        for channel in range(num_channels):
            channel_to_inpaint = image_to_inpaint[:, :, channel].copy()
            channel_mask = binary_mask
            image_flat = channel_to_inpaint.flatten()
            mask_flat = channel_mask.flatten()

            region_to_inpaint = image_flat[mask_flat == 255].reshape(-1, 1)
            gmm = GaussianMixture(n_components=3, random_state=0)
            gmm.fit(region_to_inpaint)
            background_probs = gmm.predict_proba(region_to_inpaint)[:, 0]  # Probability of being bleed-through
            
            inpainted_channel = inpainted_image[:, :, channel].copy()
            inpainted_channel = inpainted_channel * 255
            inpainted_channel = inpainted_channel.astype(np.uint8)
            bleed_through_mask_2d = build_mask_from_probs(background_probs, channel_mask, inpainted_channel)
            inpainted_channel= np.where(bleed_through_mask_2d, np.median(inpainted_channel), inpainted_channel) 
            inpainted_image[:, :, channel] = inpainted_channel
        inpainted_image = inpainted_image.astype(np.uint8)

        if 'HSV' in text_model_name:
            inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_HSV2RGB)
        elif 'LAB' in text_model_name:
            x_HSV = inpainted_image[:,:,-1]
            x_LAB = inpainted_image[:,:,:3]
            inpainted_image = cv2.cvtColor(x_LAB, cv2.COLOR_LAB2RGB)

        return inpainted_image
    
    def biweight_image_inpainting(self,
                                page_extraction_model_name='Residual_attention_UNet_page_extraction',
                                ornament_model_name='Residual_attention_UNet_ornament_extraction',
                                text_model_name='Residual_attention_UNet_text_extraction',
                                sigma_clip_sigma=None,
                                sigma_clip_sigma_lower=None,
                                sigma_clip_sigma_upper=None,
                                sigma_clip_maxiters=None,
                                save_folder_path_mask_page = None):
        
        '''
        This function leverages the biweight location estimator to inpaint the bleed-through regions of the image.
        Biweight is a robust statistical estimator used to measure the central tendency and the scale of a distribution. 
        It is robust because it down-weights the contribution of extreme values, allowing to provide a more representative
        measure of the central tendency.
        This function calculates the biweight location of the areas of page_filtered_image where the mask is 1, and then
        uses the sigma_clip function from astropy to distinguish between the inliers and the outliers. The outliers are then
        inpainted using the biweight location.
        ACTUALLY, THE RESULT OF sigma_clip IS NOT WHAT I HOPED. IT DOESN'T DETECT THE BLEED-THROUGH AS OUTLIER, BUT IT DETECTS
        OTHER FEATURES OF THE PICTURE AS OUTLIERS. SINCE THE BLEED-THROUGH IS THEN CONSIDERED AN INLIER LIKE THE PARTS OF THE 
        PAGES THAT RESISTED DECAY THE MOST WE CAN INVERT THE MASK RETURNED BY sigma_clip AND USE IT TO INPAINT THE BLEED-THROUGH
        AND KEEPING UNTOUCHED THE PARTS OF THE PAGE LIKE THE OUTSIDE OF THE PAGE, THE PARTS THAT RESISTED THE DECAY THE LEAST AND 
        OTHERS.
        This is done channel by channel.
        '''

        if save_folder_path_mask_page:
            # page_filtered_save_path = os.path.join(save_folder_path_mask_page, os.path.dirname(self.image_path).split('_')[0]+'_'+os.path.basename(self.image_path).split('_')[1]+'_page_filtered.png')
            # mask_bleed_through_save_path = os.path.join(save_folder_path_mask_page, os.path.dirname(self.image_path).split('_')[0]+'_'+os.path.basename(self.image_path).split('_')[1]+'_BLEED_THROUGH_MASK.png')
            page_filtered_save_path = os.path.join(save_folder_path_mask_page, os.path.basename(self.image_path).split('.')[0]+'_page_filtered.png')
            mask_bleed_through_save_path = os.path.join(save_folder_path_mask_page, os.path.basename(self.image_path).split('.')[0]+'_BLEED_THROUGH_MASK.png')
            
            if os.path.exists(page_filtered_save_path) and os.path.exists(mask_bleed_through_save_path):
                page_filtered_image = np.array(Image.open(page_filtered_save_path))
                mask = np.array(Image.open(mask_bleed_through_save_path))
            else:
                page_filtered_image, mask = self.bleed_through_finder(page_extraction_model_name=page_extraction_model_name,
                                                ornament_model_name=ornament_model_name,
                                                text_model_name=text_model_name)
                Image.fromarray(page_filtered_image).save(page_filtered_save_path)
                Image.fromarray(mask).save(mask_bleed_through_save_path)
        else:
            page_filtered_image, mask = self.bleed_through_finder(page_extraction_model_name=page_extraction_model_name,
                                            ornament_model_name=ornament_model_name,
                                            text_model_name=text_model_name)

        if 'HSV' in text_model_name:
            page_filtered_image = np.array(page_filtered_image.convert('HSV'))
        elif 'LAB' in text_model_name:
            x_HSV = np.array(page_filtered_image.convert('HSV'))
            x_LAB = cv2.cvtColor(np.array(page_filtered_image), cv2.COLOR_RGB2LAB)
            page_filtered_image = np.concatenate((x_LAB, x_HSV[:,:,2][:,:,None]), axis=2)
        else:
            page_filtered_image = np.array(page_filtered_image)


        cleaned_image = np.array(page_filtered_image).copy()

        num_channels = page_filtered_image.shape[2]
        mask = np.repeat(mask[:, :, np.newaxis], num_channels, axis=2)
        cleaned_image = cleaned_image / 255
        image_to_inpaint = cleaned_image.copy()
        # image_to_inpaint[mask == 0] = 0 # Zero out the important regions

        inpainted_image = cleaned_image.copy()

        for channel in range(num_channels):
            channel_to_inpaint = image_to_inpaint[:, :, channel].copy()
            channel_mask = mask[:, :, channel]
            image_flat = channel_to_inpaint.flatten()
            mask_flat = channel_mask.flatten()
            region_to_inpaint = image_flat[mask_flat == 255].reshape(-1, 1)
            biloc = biweight_location(region_to_inpaint, c=6.0)
            filtered_data = sigma_clip(region_to_inpaint, sigma=sigma_clip_sigma, sigma_lower=sigma_clip_sigma_lower, sigma_upper=sigma_clip_sigma_upper, maxiters=sigma_clip_maxiters)

            # The next two lines are a trick which leverage the errors of sigma_clipping
            inverted_mask = ~filtered_data.mask
            filtered_data.mask = inverted_mask

            filtered_data_filled = filtered_data.filled(biloc)
            inpainted_channel = inpainted_image[:, :, channel].copy()
            inpainted_channel[channel_mask == 255] = filtered_data_filled.flatten()

            # def plot_before_after_biweight_location(region_to_inpaint, sigma, sigma_lower, sigma_upper, maxiters):
            #     biloc = biweight_location(region_to_inpaint, c=6.0)
            #     if sigma is None:
            #         filtered_data = sigma_clip(region_to_inpaint, sigma_lower=sigma_lower,sigma_upper=sigma_upper, maxiters=maxiters)
            #     elif sigma_lower is None and sigma_upper is None:
            #         filtered_data = sigma_clip(region_to_inpaint, sigma=sigma, maxiters=maxiters)
            #     filtered_data_filled = filtered_data.filled(biloc)
            #     inpainted_channel = inpainted_image[:, :, channel].copy()
            #     inpainted_channel[channel_mask == 1] = filtered_data_filled.flatten()
            #     inverted_mask = ~filtered_data.mask
            #     filtered_data.mask = inverted_mask
            #     filtered_data_filled = filtered_data.filled(biloc)
            #     inpainted_channel_inverse = inpainted_image[:, :, channel].copy()
            #     inpainted_channel_inverse[channel_mask == 1] = filtered_data_filled.flatten()
            #     fig, axs = plt.subplots(1,3, figsize=(10,8))
            #     axs[0].imshow(inpainted_image[:,:,channel], cmap='gray')
            #     axs[0].set_title('Original Image')
            #     axs[1].imshow(inpainted_channel, cmap='gray')
            #     axs[1].set_title('Inpainted sigma_clip standard mask')
            #     axs[2].imshow(inpainted_channel_inverse, cmap='gray')
            #     axs[2].set_title('Inpaint sigma_clip inverse mask')
            #     plt.show()
            #     return inpainted_channel
                
            # inpainted_channel = plot_before_after_biweight_location(region_to_inpaint,None,1,2,5)
            # import ipdb; ipdb.set_trace()

            inpainted_image[:, :, channel] = inpainted_channel

        inpainted_image = inpainted_image*255
        inpainted_image = inpainted_image.astype(np.uint8)

        if 'HSV' in text_model_name:
            inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_HSV2RGB)
        elif 'LAB' in text_model_name:
            x_HSV = inpainted_image[:,:,-1]
            x_LAB = inpainted_image[:,:,:3]
            inpainted_image = cv2.cvtColor(x_LAB, cv2.COLOR_LAB2RGB)

        return inpainted_image
    
    def NLM_image_inpainting(self,
                            page_extraction_model_name='Residual_attention_UNet_page_extraction',
                            ornament_model_name='Residual_attention_UNet_ornament_extraction',
                            text_model_name='Residual_attention_UNet_text_extraction',
                            filter_strength=6, color_filter_strength=20, templateWindowSize=15, searchWindowSize=35,
                            save_folder_path_mask_page = None):

        '''
        This function takes the page filtered image and the bleed-through mask, then the pixels of the page filtered image where the mask is 0 are 
        inpainted with 0 values making them all black.
        Differently from some other methods like biweight or GMM, we perform the inpainting on the whole page filtered image (with the
        small adjustment of the black pixels) with the NLM denoising algorithm.
        Then, we take the pixels of page_filtered_image where the bleed-through mask is 0 and substitute them in the corresponding pixels of the
        NLM denoised image.
        The reason is that the NLM denoising makes also the ornaments and the text of the page smoother and this is not desired, so we can just
        replace the good areas of the page (where the bleed-through mask is equal to 0) with the original ones.
        This is done together for all the channels and not channel by channel.
        Notice also that cv2.fastNlMeansDenoising runs on the CPU.
        '''
        if templateWindowSize%2==0:
            raise ValueError('templateWindowSize must be odd')

        if save_folder_path_mask_page:
            # page_filtered_save_path = os.path.join(save_folder_path_mask_page, os.path.dirname(self.image_path).split('_')[0]+'_'+os.path.basename(self.image_path).split('_')[1]+'_page_filtered.png')
            # mask_bleed_through_save_path = os.path.join(save_folder_path_mask_page, os.path.dirname(self.image_path).split('_')[0]+'_'+os.path.basename(self.image_path).split('_')[1]+'_BLEED_THROUGH_MASK.png')
           
            page_filtered_save_path = os.path.join(save_folder_path_mask_page, os.path.basename(self.image_path).split('.')[0] +'_page_filtered.png')
            mask_bleed_through_save_path = os.path.join(save_folder_path_mask_page, os.path.basename(self.image_path).split('.')[0] +'_BLEED_THROUGH_MASK.png')
 
            if os.path.exists(page_filtered_save_path) and os.path.exists(mask_bleed_through_save_path):
                page_filtered_image = np.array(Image.open(page_filtered_save_path))
                mask = np.array(Image.open(mask_bleed_through_save_path))
            else:
                page_filtered_image, mask = self.bleed_through_finder(page_extraction_model_name=page_extraction_model_name,
                                ornament_model_name=ornament_model_name,
                                text_model_name=text_model_name)
                Image.fromarray(page_filtered_image).save(page_filtered_save_path)
                Image.fromarray(mask).save(mask_bleed_through_save_path)
        else:
            page_filtered_image, mask = self.bleed_through_finder(page_extraction_model_name=page_extraction_model_name,
                                            ornament_model_name=ornament_model_name,
                                            text_model_name=text_model_name)

        image_to_inpaint = page_filtered_image.copy()

        if self.device== "cuda":
            gpu_image = cv2.cuda_GpuMat()  # Create a GPU image container
            gpu_image.upload(image_to_inpaint)  # Upload image to GPU

        if color_filter_strength:
            print("NLM Color filter strength is used")
            if self.device=="cuda":
                nlm_denoised_image = cv2.fastNlMeansDenoisingColored(gpu_image, None, h=filter_strength, hColor=color_filter_strength, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
            else:
                nlm_denoised_image = cv2.fastNlMeansDenoisingColored(image_to_inpaint, None, h=filter_strength, hColor=color_filter_strength, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
        else:
            print("NLM Color filter strength is not used")
            if self.device=="cuda":
                nlm_denoised_image = cv2.fastNlMeansDenoising(gpu_image, None, h=filter_strength, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)
            else:
                nlm_denoised_image = cv2.fastNlMeansDenoising(image_to_inpaint, None, h=filter_strength, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize)

        pixels_not_to_inpaint = page_filtered_image[mask == 0]
        nlm_denoised_image[mask == 0] = pixels_not_to_inpaint
        return nlm_denoised_image
    
    def Gaussian_denoise_image_inpainting(self,
                            page_extraction_model_name='Residual_attention_UNet_page_extraction',
                            ornament_model_name='Residual_attention_UNet_ornament_extraction',
                            text_model_name='Residual_attention_UNet_text_extraction',
                            sigma=1,
                            save_folder_path_mask_page = None):

        '''
        This function takes the page filtered image and the bleed-through mask, then the pixels of the page filtered image where the mask is 0 are 
        inpainted with 0 values making them all black.
        Differently from some other methods like biweight or GMM, we perform the inpainting on the whole page filtered image (with the
        small adjustment of the black pixels) with the Gaussian denoising mask.
        Then, we take the pixels of page_filtered_image where the bleed-through mask is 0 and substitute them in the corresponding pixels of the
        NLM denoised image.
        This is done together for all the channels and not channel by channel. 
        '''

        if save_folder_path_mask_page:
            # page_filtered_save_path = os.path.join(save_folder_path_mask_page, os.path.dirname(self.image_path).split('_')[0]+'_'+os.path.basename(self.image_path).split('_')[1]+'_page_filtered.png')
            # mask_bleed_through_save_path = os.path.join(save_folder_path_mask_page, os.path.dirname(self.image_path).split('_')[0]+'_'+os.path.basename(self.image_path).split('_')[1]+'_BLEED_THROUGH_MASK.png')
            page_filtered_save_path = os.path.join(save_folder_path_mask_page, os.path.basename(self.image_path).split('.')[0]+'_page_filtered.png')
            mask_bleed_through_save_path = os.path.join(save_folder_path_mask_page, os.path.basename(self.image_path).split('.')[0]+'_BLEED_THROUGH_MASK.png')
            
            if os.path.exists(page_filtered_save_path) and os.path.exists(mask_bleed_through_save_path):
                page_filtered_image = np.array(Image.open(page_filtered_save_path))
                mask = np.array(Image.open(mask_bleed_through_save_path))
            else:
                page_filtered_image, mask = self.bleed_through_finder(page_extraction_model_name=page_extraction_model_name,
                                ornament_model_name=ornament_model_name,
                                text_model_name=text_model_name)
                Image.fromarray(page_filtered_image).save(page_filtered_save_path)
                Image.fromarray(mask).save(mask_bleed_through_save_path)
        else:
            page_filtered_image, mask = self.bleed_through_finder(page_extraction_model_name=page_extraction_model_name,
                                            ornament_model_name=ornament_model_name,
                                            text_model_name=text_model_name)

        image_to_inpaint = page_filtered_image.copy()
        
        for channel in range(3):
            image_to_inpaint_channel = image_to_inpaint[:,:,channel]

            gaussian_denoised_image = gaussian_filter(image_to_inpaint_channel, sigma=sigma)

            pixels_not_to_inpaint = image_to_inpaint_channel[mask == 0]

            gaussian_denoised_image[mask == 0] = pixels_not_to_inpaint

            image_to_inpaint[:,:,channel] = gaussian_denoised_image
        return image_to_inpaint

if __name__ == "__main__":
    # img_names = os.listdir(os.path.join('DIBCO_DATA','DIBCO2019'))
    # img_names = os.listdir(os.path.join('Bleed_Through_Database', 'rgb'))
    # img_names = os.listdir(os.path.join("4C1_PALLADIUS_FUSCUS"))
    # img_names = ['CNMD0000263308_0170_Carta_82v.jpg', 'CNMD0000263308_0111_Carta_53r.jpg', 'CNMD0000263308_0068_Carta_31v.jpg', "CNMD0000263308_0278_Carta_136v.jpg", "CNMD0000263308_0288_Carta_141v.jpg"]
    img_names = ['CNMD0000263308_0090_Carta_42v.jpg']

    for img_name in img_names:
        # image_path = os.path.join('DIBCO_DATA','DIBCO2019',img_name)
        # image_path = os.path.join(os.path.join('Bleed_Through_Database', 'rgb', img_name))
        image_path = os.path.join("Napoli_Biblioteca_dei_Girolamini_CF_2_16_Filippino", img_name)
        # image_path = os.path.join("Firenze_BibliotecaMediceaLaurenziana_Plut_40_1", img_name)
        # image_path = os.path.join("4C1_PALLADIUS_FUSCUS", img_name)

        models_folder_path = 'models_CHECKING_2019'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'mps'
        print('Using device:', device)

        cleaner = bleed_through_cleaner(image_path, models_folder_path, device)

        save_folder_path = 'TO_REMOVE_folder'
        # save_folder_path = 'assets'
        # save_folder_path = os.path.join('TO_REMOVE_folder', 'cleaned_rgb')
        # save_folder_path = '4C1_PALLADIUS_FUSCUS_cleaned'
        # save_folder_path = os.path.join("DIBCO_evaluation", "DIBCO_DATA_until_2019_pred_400_epochs", "Images", "DIBCO2019")

        mask_page_folder_path = save_folder_path # Use None if you don't want to save the mask and the page
        os.makedirs(save_folder_path, exist_ok=True)

        ornament_model_name = "Residual_attention_UNet_ornament_extraction"
        # ornament_model_name = "Residual_attention_UNet_ornament_extraction_finetuning"
        text_model_name = "Residual_attention_UNet_text_extraction_finetuning_400_epochs"
        # text_model_name = "Residual_attention_UNet_text_extraction"
        # page_extraction_model_name = None
        page_extraction_model_name = "Residual_attention_UNet_page_extraction"

        # cleaned_image = cleaner.median_image_inpainting(save_folder_path_mask_page = mask_page_folder_path,
        #                                                             ornament_model_name=ornament_model_name,
        #                                                             text_model_name=text_model_name,
        #                                                             page_extraction_model_name = page_extraction_model_name
        #                                                             )
        # extension = img_name.split('.')[-1]
        # new_name = img_name.replace('.'+extension,'')+'_median.png'
        # Image.fromarray(cleaned_image).save(os.path.join(save_folder_path, new_name))

        cleaned_image = cleaner.biweight_image_inpainting(ornament_model_name=ornament_model_name,
                                                                   text_model_name=text_model_name,
                                                                   page_extraction_model_name = page_extraction_model_name,
                                                                        sigma_clip_sigma=None, sigma_clip_sigma_lower=1.5, 
                                                                        sigma_clip_sigma_upper=2, sigma_clip_maxiters=3,
                                                                        save_folder_path_mask_page = mask_page_folder_path)
        extension = img_name.split('.')[-1]
        new_name = img_name.replace('.'+extension,'')+'_biweight.png'
        Image.fromarray(cleaned_image).save(os.path.join(save_folder_path, new_name))


        # cleaned_image = cleaner.GMM_image_inpainting(ornament_model_name=ornament_model_name,
        #                                                            text_model_name=text_model_name,
        #                                                            page_extraction_model_name=page_extraction_model_name,
        #                                                             save_folder_path_mask_page = mask_page_folder_path)
        # extension = img_name.split('.')[-1]
        # new_name = img_name.replace('.'+extension,'')+'_GMM.png'
        # Image.fromarray(cleaned_image).save(os.path.join(save_folder_path, new_name))

        # for filter_strength in [5,6,7,8,9,10]:
        #     for color_filter_strength in [10,20,30]:
        #         for templateWindowSize in [11, 15, 21]:
        #             for searchWindowSize in [35, 45, 55]:
        #                 cleaned_image = cleaner.NLM_image_inpainting(ornament_model_name=ornament_model_name,
        #                                                                         text_model_name=text_model_name,
        #                                                                         page_extraction_model_name=page_extraction_model_name,
        #                                                                             filter_strength=filter_strength, color_filter_strength=color_filter_strength, templateWindowSize=templateWindowSize, searchWindowSize=searchWindowSize,
        #                                                                         save_folder_path_mask_page = mask_page_folder_path)
        #                 # new_name = img_folder.split('_')[0]+'_'+img_name.split('_')[1]+'_NLM.png'
        #                 new_name = f'{img_name}_NLM_{filter_strength}_{color_filter_strength}_{templateWindowSize}_{searchWindowSize}.png'
        #                 Image.fromarray(cleaned_image).save(os.path.join(save_folder_path, new_name))
    
        num_smoothings = 1
            
        cleaned_image = cleaner.NLM_image_inpainting(ornament_model_name=ornament_model_name,
                                                        text_model_name=text_model_name,
                                                        page_extraction_model_name=page_extraction_model_name,
                                                            # filter_strength=6, color_filter_strength=20, templateWindowSize=15, searchWindowSize=35,
                                                            filter_strength=8, color_filter_strength=20, templateWindowSize=25, searchWindowSize=55,
                                                        save_folder_path_mask_page = mask_page_folder_path)

        extension = img_name.split('.')[-1]
        new_name = img_name.replace('.'+extension,'')+'_NLM_strong.png'
        Image.fromarray(cleaned_image).save(os.path.join(save_folder_path, new_name))
        
        if num_smoothings > 1:
            for _ in range(num_smoothings):
                os.rename(os.path.join(save_folder_path, new_name), os.path.join(save_folder_path, new_name.replace('_NLM.png', '.png')))
                os.rename(os.path.join(save_folder_path, new_name.replace('_NLM.png', '_page_filtered.png')), os.path.join(save_folder_path, new_name.replace('_NLM.png', '_page_filtered_old.png')))
                shutil.copy(os.path.join(save_folder_path, new_name.replace('_NLM.png', '.png')), os.path.join(save_folder_path, new_name.replace('_NLM.png', '_page_filtered.png')))
                image_path  = os.path.join(save_folder_path, new_name.replace('_NLM.png', '.png'))
                cleaner = bleed_through_cleaner(image_path, models_folder_path, device)
                cleaned_image = cleaner.NLM_image_inpainting(ornament_model_name=ornament_model_name,
                                                            text_model_name=text_model_name,
                                                            page_extraction_model_name=page_extraction_model_name,
                                                                # filter_strength=6, color_filter_strength=20, templateWindowSize=15, searchWindowSize=35,
                                                                filter_strength=8, color_filter_strength=20, templateWindowSize=25, searchWindowSize=55,
                                                            save_folder_path_mask_page = mask_page_folder_path)
                Image.fromarray(cleaned_image).save(os.path.join(save_folder_path, new_name))
            os.remove(os.path.join(save_folder_path, new_name.replace('_NLM.png', '.png')))

        # cleaned_image = cleaner.Gaussian_denoise_image_inpainting(sigma=5, save_folder_path_mask_page = mask_page_folder_path)
        # extension = img_name.split('.')[-1]
        # new_name = img_name.replace('.'+extension,'')+'_GaussDenoise.png'
        # Image.fromarray(cleaned_image).save(os.path.join(save_folder_path, new_name))





     


