#!/bin/bash

read -p "Enter the base directory: " BASE_DIR
# you have to write D:/bleed-through-cleaner in the prompt

DIBCO_year=2018

image_dir="$BASE_DIR/DIBCO_DATA/DIBCO"$DIBCO_year"_GT"
output_dir="$BASE_DIR/DIBCO_evaluation/DIBCO_DATA_until_2019_pred_400_epochs/Weights/DIBCO"$DIBCO_year""

# Ensure the output directory exists
mkdir -p "$output_dir"

# Iterate through ground truth images (assuming GT filenames contain '_GT')
for gt_file in "$image_dir"/*.bmp; do
    # Extract base filename without extension
    base_name=$(basename "$gt_file" ".bmp")

    # Expected output filenames (saved in the same directory as input images)
    r_weights="${base_name}_RWeights.dat"
    p_weights="${base_name}_PWeights.dat"

    # Run BinEvalWeights.exe with correct parameters
    ./BinEvalWeights.exe "$gt_file" "$r_weights" "$p_weights"
    
    mv "$image_dir/$r_weights"  "$output_dir/"
    mv "$image_dir/$p_weights"  "$output_dir/"
done
