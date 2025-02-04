#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate MAGIC

model_name="Residual_attention_UNet_text_extraction_finetuning_400_epochs"
snapshot_folder_path="models_CHECKING_2019"
# dataset_path="DIBCO_DATA_patches_ALL"
dataset_path="DIBCO_DATA_patches_until_2019"

# img_path="READ-BAD/img-CS18/img/public-test/e-codices_csg-0018_098_max.jpg"
# img_path="Firenze_BibliotecaMediceaLaurenziana_Plut_40_1/CNMD0000250043_0676_Carta_333r.jpg"
# destination_path="baseline_mask.png"

# python3 training.py --multiple_gpus=False --save_every=10 --snapshot_folder_path="$snapshot_folder_path"  --dataset_path="$dataset_path" --batch_size=6 --model_name="$model_name" --lr=2e-4 --num_epochs=501 --out_dim=1
# python training.py --multiple_gpus=False --save_every=10 --snapshot_folder_path="models"  --dataset_path="dataset_text_extraction_patches" --batch_size=6 --model_name="TO_REMOVE" --lr=2e-4 --num_epochs=501 --out_dim=1

# python3 Aggregation_Sampling.py --snapshot_folder_path="$snapshot_folder_path" --model_name="$model_name" --patch_size=256 --stride=64 --img_path="$img_path" --destination_path="$destination_path" --out_dim=1
# python Aggregation_Sampling.py --snapshot_folder_path="models" --model_name="Residual_attention_UNet_ornament_extraction" --patch_size=256 --stride=64 --img_path="Napoli_Biblioteca_dei_Girolamini_CF_2_16(Filippino)/CNMD0000263308_0174_Carta_84v.jpg" --destination_path="ornament_mask.png" --out_dim=1

python training.py --multiple_gpus=False --save_every=10 --snapshot_folder_path="$snapshot_folder_path"  --dataset_path="$dataset_path" --batch_size=16 --model_name="$model_name" --lr=2e-4 --num_epochs=901 --out_dim=1
