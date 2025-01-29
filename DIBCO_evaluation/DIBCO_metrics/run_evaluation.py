import os
import subprocess
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define absolute paths
exe_path = os.path.abspath("DIBCO_metrics.exe")

gt_image = os.path.abspath("PR_GT.png")
binarized_image = os.path.abspath("PR_bin.png")

recall_weight = "PR_RWeights.dat"
precision_weight = "PR_PWeights.dat"

# img_number = 7
# DIBCO_year = 2013
# if DIBCO_year == 2013:
#     img_number = "PR0" + str(img_number)

# gt_image = os.path.join('..','..','DIBCO_DATA', f'DIBCO{DIBCO_year}_GT', f'{img_number}_gt.bmp')
# binarized_image = os.path.join('..', 'DIBCO_DATA_pred', 'Images', f'DIBCO{DIBCO_year}', f'{img_number}_BLEED_THROUGH_MASK.png')

# recall_weight = os.path.join('..', 'DIBCO_DATA_pred', 'Weights', f'DIBCO{DIBCO_year}', f'{img_number}_gt_RWeights.dat')
# precision_weight = os.path.join('..', 'DIBCO_DATA_pred', 'Weights', f'DIBCO{DIBCO_year}', f'{img_number}_gt_PWeights.dat')


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
