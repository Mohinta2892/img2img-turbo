# Pix2Pix
## ValueError: Attempting to unscale FP16 gradients.

```
Traceback (most recent call last):██████████████████████████████████████████████████████████▎                                           | 2/3 [00:08<00:05,  5.09s/it]
  File "/cephfs/smohinta/img2img-turbo/src/train_pix2pix_turbo.py", line 307, in <module>
    main(args)
  File "/cephfs/smohinta/img2img-turbo/src/train_pix2pix_turbo.py", line 190, in main
    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
  File "/lmb/home/smohinta/anaconda3/envs/img2img-turbo/lib/python3.10/site-packages/accelerate/accelerator.py", line 2157, in clip_grad_norm_
    self.unscale_gradients()
  File "/lmb/home/smohinta/anaconda3/envs/img2img-turbo/lib/python3.10/site-packages/accelerate/accelerator.py", line 2107, in unscale_gradients
    self.scaler.unscale_(opt)
  File "/lmb/home/smohinta/anaconda3/envs/img2img-turbo/lib/python3.10/site-packages/torch/amp/grad_scaler.py", line 338, in unscale_
    optimizer_state["found_inf_per_device"] = self._unscale_grads_(
  File "/lmb/home/smohinta/anaconda3/envs/img2img-turbo/lib/python3.10/site-packages/torch/amp/grad_scaler.py", line 260, in _unscale_grads_
    raise ValueError("Attempting to unscale FP16 gradients.")
ValueError: Attempting to unscale FP16 gradients.

```

## Solution
We cannot finetune LORA with fp16, run the pix2pix with ```--mixed_precision "no"```
```
accelerate launch src/train_pix2pix_turbo.py --pretrained_model_name_or_path="stabilityai/sd-turbo" \
--output_dir="output/pix2pix_turbo/em_hemi_boundaries_to_raw_pix2pix" \
--dataset_folder="data/em_hemi_boundaries_to_raw_pix2pix" --resolution=256 \
--train_batch_size=4 --enable_xformers_memory_efficient_attention \
--viz_freq 25 --track_val_fid --report_to "wandb" \
--tracker_project_name "pix2pix_turbo_b2h" --mixed_precision "no"
```
