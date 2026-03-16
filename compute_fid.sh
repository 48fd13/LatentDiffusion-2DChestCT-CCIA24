
REAL_IMG_DIR="data/train_data/nodules_6mm_512x512"
REAL_MASKS_DIR="data/train_data/masks_6mm_512x512_sq"


# MODEL 1 - Self-attention with localization mask
log_id=2024-07-13_10-10-26_FINAL_CCIA24_model1_selfattention_locmask
FAKE_IMG_DIR="data/ckpts/$log_id/fid_synthetic/ims"
FAKE_MASKS_DIR="data/ckpts/$log_id/fid_synthetic/masks"
python3 compute_fid.py --real_images_dir $REAL_IMG_DIR --synthetic_images_dir $FAKE_IMG_DIR \
                       --nodule_level --real_masks_dir $REAL_MASKS_DIR --synthetic_masks_dir $FAKE_MASKS_DIR


# MODEL 2 - Cross-attention with localization mask
log_id=2024-07-12_10-03-18_FINAL_CCIA24_model2_crossattention_locmask_maskv2
FAKE_IMG_DIR="data/ckpts/$log_id/fid_synthetic/ims"
FAKE_MASKS_DIR="data/ckpts/$log_id/fid_synthetic/masks"
python3 compute_fid.py --real_images_dir $REAL_IMG_DIR --synthetic_images_dir $FAKE_IMG_DIR \
                       --nodule_level --real_masks_dir $REAL_MASKS_DIR --synthetic_masks_dir $FAKE_MASKS_DIR


# MODEL 3 - Cross-attention with localization mask + nodule attribute embeddings
log_id=2024-07-12_19-57-59_FINAL_CCIA24_model3_crossattention_locmask_noduleattributes_maskv2
FAKE_IMG_DIR="data/ckpts/$log_id/fid_synthetic/ims"
FAKE_MASKS_DIR="data/ckpts/$log_id/fid_synthetic/masks"
python3 compute_fid.py --real_images_dir $REAL_IMG_DIR --synthetic_images_dir $FAKE_IMG_DIR \
                       --nodule_level --real_masks_dir $REAL_MASKS_DIR --synthetic_masks_dir $FAKE_MASKS_DIR