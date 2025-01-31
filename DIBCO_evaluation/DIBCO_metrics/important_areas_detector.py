#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_weights(file_path):
    # Read the .dat file as an array
    return np.loadtxt(file_path)

DIBCO_year = 2014
img_number = 7

if DIBCO_year == 2014:
    # img_number = "PR0" + str(img_number)
    if img_number < 10:
        img_number = "H0" + str(img_number)
    else:
        img_number = "H" + str(img_number)

# Replace with the path to your Precision and Recall weights
precision_weights_path = os.path.join('..', 'DIBCO_DATA_pred', 'Weights', 'DIBCO'+str(DIBCO_year), str(img_number)+'_PWeights.dat')
recall_weights_path =  os.path.join('..', 'DIBCO_DATA_pred', 'Weights', 'DIBCO'+str(DIBCO_year), str(img_number)+'_RWeights.dat')

precision_weights = load_weights(precision_weights_path)
recall_weights = load_weights(recall_weights_path)

# Check the shape of the loaded weights (they should match the image size)
print("Precision Weights Shape:", precision_weights.shape)
print("Recall Weights Shape:", recall_weights.shape)

# %%
def normalize_weights(weights):
    # Normalize weights to the range [0, 255]
    return np.uint8(255 * (weights - np.min(weights)) / (np.max(weights) - np.min(weights)))

normalized_precision = normalize_weights(precision_weights)
normalized_recall = normalize_weights(recall_weights)
#%%
image = np.array(Image.open(os.path.join('..', 'DIBCO_DATA_pred', 'Images', 'DIBCO'+str(DIBCO_year), str(img_number)+'_BLEED_THROUGH_MASK.png')))
height, width = image.shape[:2]
normalized_precision = normalized_precision.reshape((height, width))
normalized_recall = normalized_recall.reshape((height, width))
# %%
def plot_heatmap(weights, title="Heatmap"):
    plt.imshow(weights, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

# Display heatmaps for precision and recall weights
plot_heatmap(normalized_precision, title="Precision Weight Heatmap")
plot_heatmap(normalized_recall, title="Recall Weight Heatmap")
#%%
