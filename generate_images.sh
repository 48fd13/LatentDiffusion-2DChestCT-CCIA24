
MASKS_DIR="data/train_data/masks_6mm_512x512_sq"

:'
# MODEL 0 - Unconditional synthesis
log_id=2024-06-12_18-01-52_FINAL_CCIA24_unconditional_latent_BS8
CKPT_DIR="data/ckpts/$log_id"
OUT_DIR="data/outputs/$log_id"
python3 generate_N_images.py \
	--ckpt_dir $CKPT_DIR \
	--out_dir $OUT_DIR \
	--n_images 4 \
	--batch_size 4 \
	--overwrite


# MODEL 1 - Self-attention with localization mask
log_id=2024-07-13_10-10-26_FINAL_CCIA24_model1_selfattention_locmask
CKPT_DIR="data/ckpts/$log_id"
OUT_DIR="data/outputs/$log_id"
python3 generate_N_images.py \
	--ckpt_dir $CKPT_DIR \
	--out_dir $OUT_DIR \
	--masks_dir $MASKS_DIR \
	--n_images 4 \
	--batch_size 4 \
	--overwrite
'
# MODEL 2 - Cross-attention with localization mask
log_id=2024-07-12_10-03-18_FINAL_CCIA24_model2_crossattention_locmask_maskv2
CKPT_DIR="data/ckpts/$log_id"
OUT_DIR="data/outputs/$log_id"
python3 generate_N_images.py \
	--ckpt_dir $CKPT_DIR \
	--out_dir $OUT_DIR \
	--masks_dir $MASKS_DIR \
	--n_images 4 \
	--batch_size 4 \
	--overwrite

# MODEL 3 - Cross-attention with localization mask + nodule attribute embeddings
log_id=2024-07-12_19-57-59_FINAL_CCIA24_model3_crossattention_locmask_noduleattributes_maskv2
CKPT_DIR="data/ckpts/$log_id"
OUT_DIR="data/outputs/$log_id"
python3 generate_N_images.py \
	--ckpt_dir $CKPT_DIR \
	--out_dir $OUT_DIR \
	--masks_dir $MASKS_DIR \
	--n_images 4 \
	--batch_size 4 \
	--overwrite