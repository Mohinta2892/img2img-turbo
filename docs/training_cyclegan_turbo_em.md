# Data Download
  - TODO

# Train

```
accelerate launch --main_process_port 29501 src/train_cyclegan_turbo.py \
  --pretrained_model_name_or_path="stabilityai/sd-turbo"  \
  --output_dir="output/cyclegan_turbo/em_hemi_octo"  --dataset_folder "data/em_source_target" \
  --train_img_prep "resize_256_randomcrop_200x200_hflip" --val_img_prep "no_resize"
  --learning_rate="1e-5" --max_train_steps=25000  --train_batch_size=1 --gradient_accumulation_steps=1
  --report_to "wandb" --tracker_project_name "mohinta_unpaired_em_h2o_cycle_debug_v1" \
  --enable_xformers_memory_efficient_attention \
  --validation_steps 5000 --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1
```
# Inference

**a2b**:
```
python src/inference_unpaired.py --model_path "output/cyclegan_turbo/em_hemi_octo/checkpoints/model_25001.pkl" \
    --input_image "data/em_hemi_octo/test_A/img_596.png" \
    --prompt "electron microscopy image from the hemibrain of the female adult fruit fly" --direction "a2b" \
    --output_dir "outputs" --image_prep "no_resize"

```

**b2a**:
```
python src/inference_unpaired.py --model_path "output/cyclegan_turbo/em_hemi_octo/checkpoints/model_25001.pkl" \
    --input_image "data/em_hemi_octo/test_B/img_461.png" \
    --prompt "electron microscopy image from larval brain of a fly” --direction "b2a” \
    --output_dir "outputs" --image_prep "no_resize"

