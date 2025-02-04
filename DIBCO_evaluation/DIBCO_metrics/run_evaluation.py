
######## HERE ON THERE IS THE CODE TO RUN THE EVALUATION FOR A SINGLE IMAGE #########

# import os
# import subprocess
# import cv2
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # Define absolute paths
# exe_path = os.path.abspath("DIBCO_metrics.exe")

# img_number = 1
# DIBCO_year = 2017

# if DIBCO_year == 2014:
#     # img_number = "PR0" + str(img_number)
#     if img_number < 10:
#         img_number = "H0" + str(img_number)
#     else:
#         img_number = "H" + str(img_number)

# gt_image = os.path.join('..','..','DIBCO_DATA', f'DIBCO{DIBCO_year}_GT', f'{img_number}.bmp')
# binarized_image = os.path.join('..', 'DIBCO_DATA_pred', 'Images', f'DIBCO{DIBCO_year}', f'{img_number}_BLEED_THROUGH_MASK.png')

# recall_weight = os.path.join('..', 'DIBCO_DATA_pred', 'Weights', f'DIBCO{DIBCO_year}', f'{img_number}_RWeights.dat')
# precision_weight = os.path.join('..', 'DIBCO_DATA_pred', 'Weights', f'DIBCO{DIBCO_year}', f'{img_number}_PWeights.dat')


# # Print paths to verify correctness
# print(f"Executable Path: {exe_path}")
# print(f"Ground Truth Image Path: {gt_image}")
# print(f"Binarized Image Path: {binarized_image}")

# # Check if files exist
# for file in [exe_path, gt_image, binarized_image]:
#     if not os.path.exists(file):
#         raise FileNotFoundError(f"Error: File not found -> {file}")

# # Run the executable with absolute paths
# try:
#     result = subprocess.run([exe_path, gt_image, binarized_image, recall_weight, precision_weight], check=True)
#     print("✅ DIBCO_metrics executed successfully!")
# except subprocess.CalledProcessError as e:
#     print(f"❌ Execution failed: {e}")


######### HERE ON THERE IS THE CODE TO RUN THE EVALUATION FOR ALL IMAGES IN THE DATASET #########
import os
import subprocess
import pandas as pd

# Define absolute paths
exe_path = os.path.abspath("DIBCO_metrics.exe")
DIBCO_year = 2019

# Define paths for dataset
gt_folder = os.path.abspath(os.path.join('..', '..', 'DIBCO_DATA', f'DIBCO{DIBCO_year}_GT'))
binarized_folder = os.path.abspath(os.path.join('..', 'DIBCO_DATA_until_2019_pred_400_epochs', 'Images', f'DIBCO{DIBCO_year}'))
recall_weight_folder = os.path.abspath(os.path.join('..', 'DIBCO_DATA_until_2019_pred_400_epochs', 'Weights', f'DIBCO{DIBCO_year}'))
precision_weight_folder = os.path.abspath(os.path.join('..', 'DIBCO_DATA_until_2019_pred_400_epochs', 'Weights', f'DIBCO{DIBCO_year}'))

# Get all ground truth images (assuming they are BMP files)
gt_images = [f for f in os.listdir(gt_folder) if f.endswith('.bmp')]

# Initialize DataFrame to store results
columns = ["Image", "F-Measure", "Pseudo-F-Measure", "PSNR", "DRD", "Recall", "Pseudo-Recall", "Precision", "Pseudo-Precision"]
results_df = pd.DataFrame(columns=columns)

# Iterate over all images
for gt_filename in gt_images:
    img_number = os.path.splitext(gt_filename)[0]  # Extract filename without extension
    
    # Construct file paths
    gt_image = os.path.join(gt_folder, gt_filename)
    binarized_image = os.path.join(binarized_folder, f"{img_number}_BLEED_THROUGH_MASK.png")
    recall_weight = os.path.join(recall_weight_folder, f"{img_number}_RWeights.dat")
    precision_weight = os.path.join(precision_weight_folder, f"{img_number}_PWeights.dat")

    # Check if all necessary files exist
    if not all(os.path.exists(f) for f in [gt_image, binarized_image, recall_weight, precision_weight]):
        print(f"⚠️ Missing file(s) for image: {img_number}. Skipping...")
        continue

    # Run the executable and capture output
    try:
        result = subprocess.run(
            [exe_path, gt_image, binarized_image, recall_weight, precision_weight],
            capture_output=True, text=True, check=True
        )

        # Assuming the executable prints results in a standard format, parse the output
        output_lines = result.stdout.split("\n")
        
        # Extract statistics (adjust parsing as per actual output format)
        
        f_measure = float(output_lines[11].split('\t')[-1])
        pseudo_f_measure = float(output_lines[12].split('\t')[-1])
        psnr = float(output_lines[13].split('\t')[-1])
        drd = float(output_lines[14].split('\t')[-1])
        recall = float(output_lines[15].split('\t')[-1])
        precision = float(output_lines[16].split('\t')[-1])
        pseudo_precision = float(output_lines[18].split('\t')[-1])
        pseudo_recall = float(output_lines[17].split('\t')[-1])

        # Append results to DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([[img_number, f_measure, pseudo_f_measure, psnr, drd, recall, pseudo_recall, precision, pseudo_precision]], columns=columns)], ignore_index=True)

        print(f"✅ Processed {img_number} successfully.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {img_number}: {e}")

# Save results to CSV
output_csv = f"DIBCO{DIBCO_year}_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"📂 Results saved to {output_csv}")
