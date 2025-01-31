import os
import subprocess
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define absolute paths
exe_path = os.path.abspath("DIBCO_metrics.exe")

img_number = 7
DIBCO_year = 2014
if DIBCO_year == 2014:
    # img_number = "PR0" + str(img_number)
    if img_number < 10:
        img_number = "H0" + str(img_number)
    else:
        img_number = "H" + str(img_number)

gt_image = os.path.join('..','..','DIBCO_DATA', f'DIBCO{DIBCO_year}_GT', f'{img_number}.bmp')
binarized_image = os.path.join('..', 'DIBCO_DATA_pred', 'Images', f'DIBCO{DIBCO_year}', f'{img_number}_BLEED_THROUGH_MASK.png')

recall_weight = os.path.join('..', 'DIBCO_DATA_pred', 'Weights', f'DIBCO{DIBCO_year}', f'{img_number}_RWeights.dat')
precision_weight = os.path.join('..', 'DIBCO_DATA_pred', 'Weights', f'DIBCO{DIBCO_year}', f'{img_number}_PWeights.dat')


# Print paths to verify correctness
print(f"Executable Path: {exe_path}")
print(f"Ground Truth Image Path: {gt_image}")
print(f"Binarized Image Path: {binarized_image}")

# Check if files exist
for file in [exe_path, gt_image, binarized_image]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Error: File not found -> {file}")

# Run the executable with absolute paths
try:
    result = subprocess.run([exe_path, gt_image, binarized_image, recall_weight, precision_weight], check=True)
    print("✅ DIBCO_metrics executed successfully!")
except subprocess.CalledProcessError as e:
    print(f"❌ Execution failed: {e}")
