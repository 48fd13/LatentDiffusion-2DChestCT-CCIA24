exp_id="unconditional_latent_BS8"
dataset_name="/home/ubuntu/projects/phase-iv-ai/nodule-crops-v8/nodules_png4"
output_dir="/media/share/Datasets/diffusers_out/$exp_id"
logs_dir="/media/share/Datasets/diffusers_out/$exp_id"

accelerate launch train.py \
  --dataset_name=$dataset_name \
  --resolution=256 \
  --output_dir="/media/share/Datasets/diffusers_out/$exp_id" \
  --logging_dir=$logs_dir \
  --train_batch_size=8 \
  --num_epochs=200 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --checkpointing_steps=5000 \
  --mixed_precision=no \
  --save_images_epochs=1 \
  --save_model_epochs=1 
