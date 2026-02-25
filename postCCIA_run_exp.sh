dataset_name="/home/ubuntu/projects/phase-iv-ai/ldm-chestCT-CCIA24/paper_dataset_all/nodules_6mm_512x512"
output_dir="/media/share/Datasets/diffusers_out/$exp_id"
logs_dir="/media/share/Datasets/diffusers_out/$exp_id"
num_epochs=605
num_inference_steps=1000
bsz=4
save_freq=20 # in epochs


# MODEL 2 - Cross-attention with localization mask
exp_id="postCCIA24_crossattention_locmask_mask"
accelerate launch postCCIA_train_lidc.py \
  --dataset_name=$dataset_name \
  --resolution=512 \
  --output_dir="/media/share/Datasets/diffusers_out/$exp_id" \
  --logging_dir=$logs_dir \
  --train_batch_size=$bsz \
  --num_epochs=$num_epochs \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --checkpointing_steps=5000 \
  --save_images_epochs=$save_freq \
  --save_model_epochs=$save_freq \
  --ddpm_num_inference_steps=$num_inference_steps \
  --mixed_precision=no


