dataset_name="/home/ubuntu/projects/phase-iv-ai/ldm-chestCT-CCIA24/paper_dataset_all/nodules_6mm_512x512"
output_dir="/media/share/Datasets/diffusers_out/$exp_id"
logs_dir="/media/share/Datasets/diffusers_out/$exp_id"
num_epochs=305
num_inference_steps=1000
bsz=16
save_freq=20 # in epochs

# MODEL 2 - Cross-attention with localization mask
exp_id="FINAL_CCIA24_model2_crossattention_locmask_maskv2"
accelerate launch train_lidc.py \
  --dataset_name=$dataset_name \
  --resolution=256 \
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

exit

# MODEL 3 - Cross-attention with localization mask + nodule attribute embeddings
exp_id="FINAL_CCIA24_model3_crossattention_locmask_noduleattributes_maskv2"
accelerate launch train_lidc.py \
  --dataset_name=$dataset_name \
  --resolution=256 \
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
  --mixed_precision=no \
  --nodule_attributes


# MODEL 1 - Self-attention with localization mask
exp_id="FINAL_CCIA24_model1_selfattention_locmask"
accelerate launch train_lidc.py \
  --dataset_name=$dataset_name \
  --resolution=256 \
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
  --mixed_precision=no \
  --self_attention






