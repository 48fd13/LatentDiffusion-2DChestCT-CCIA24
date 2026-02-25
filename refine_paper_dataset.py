import glob
import numpy as np
import os
import pylidc as pl
from pylidc.utils import consensus
from PIL import Image
from tqdm import tqdm
import json
import scipy.ndimage

from generate_paper_dataset import read_jsonl, write_jsonl, update_jsonl, array2string

def get_squared_bounding_boxes(binary_mask_):

    binary_mask = (binary_mask_ > 0).astype(np.uint8)

    # Ensure the input is a binary mask
    assert binary_mask.ndim == 2, "Input mask must be a 2D array"
    assert np.issubdtype(binary_mask.dtype, np.bool_) or np.issubdtype(binary_mask.dtype, np.uint8), "Input mask must be of boolean or uint8 type"

    # Label the connected components in the binary mask
    labeled_mask, num_features = scipy.ndimage.label(binary_mask, structure=np.ones((3,3)))
    
    # Create a new binary mask of the same size as the input mask
    new_mask = np.zeros_like(binary_mask)
    
    # Find the bounding box coordinates of all connected components
    component_slices = scipy.ndimage.find_objects(labeled_mask)
    sq_sizes = []
    
    # Loop over all components to find the bounding box and create a square bounding box
    for bbox in component_slices:
        top_left_y = bbox[0].start
        bottom_right_y = bbox[0].stop
        top_left_x = bbox[1].start
        bottom_right_x = bbox[1].stop

        # Calculate the side length of the minimum enclosing square
        bbox_height = bottom_right_y - top_left_y
        bbox_width = bottom_right_x - top_left_x
        square_size = max(bbox_height, bbox_width)
        
        # Calculate the center of the bounding box
        center_y = (top_left_y + bottom_right_y) // 2
        center_x = (top_left_x + bottom_right_x) // 2
        
        half_square_size = square_size // 2
        
        if square_size > 1:
            sq_sizes.append(square_size)

            # Calculate the square boundaries ensuring they stay within the image dimensions
            square_top = max(center_y - half_square_size, 0)
            square_bottom = min(center_y + half_square_size + 1, binary_mask.shape[0])
            square_left = max(center_x - half_square_size, 0)
            square_right = min(center_x + half_square_size + 1, binary_mask.shape[1])

            # Fill the square region in the new mask
            new_mask[square_top:square_bottom, square_left:square_right] = binary_mask_[square_top:square_bottom, square_left:square_right].max()
    
    return new_mask, sq_sizes

def crop_center(array, crop_height, crop_width):
    # Get the shape of the array
    height, width = array.shape[:2]
    
    # Calculate the starting points for the crop
    start_x = width // 2 - (crop_width // 2)
    start_y = height // 2 - (crop_height // 2)
    
    # Crop and return the center of the array
    return array[start_y:start_y + crop_height, start_x:start_x + crop_width], start_y, start_x 

# this script takes the output of generate_paper_dataset.py and performs the following actions
# 	(1) crop all images to preserve only the 512x512 central part
#	(2) assign a numerical ID to each image (field 0ID in metadata.jsonl)
#       (3) update the bbx field in metadata.jsonl so it is expresesed in 6mm res instead of the original res
#	(3) create a set of masks with squared connected components

CROP_SIZE = 512
metadata_path = "/home/ubuntu/projects/phase-iv-ai/ldm-chestCT-CCIA24/paper_dataset_all/nodules_6mm/metadata.jsonl"

img_out_dir = f"/home/ubuntu/projects/phase-iv-ai/ldm-chestCT-CCIA24/paper_dataset_all/nodules_6mm_{CROP_SIZE}x{CROP_SIZE}"
metadata2_path = f"/home/ubuntu/projects/phase-iv-ai/ldm-chestCT-CCIA24/paper_dataset_all/nodules_6mm_{CROP_SIZE}x{CROP_SIZE}/metadata.jsonl"
mask_out_dir = f"/home/ubuntu/projects/phase-iv-ai/ldm-chestCT-CCIA24/paper_dataset_all/masks_6mm_{CROP_SIZE}x{CROP_SIZE}"
mask_out_dir2 = f"/home/ubuntu/projects/phase-iv-ai/ldm-chestCT-CCIA24/paper_dataset_all/masks_6mm_{CROP_SIZE}x{CROP_SIZE}_sq"

os.makedirs(img_out_dir, exist_ok=True)
os.makedirs(mask_out_dir, exist_ok=True)
os.makedirs(mask_out_dir2, exist_ok=True)

count = 0
metadata2 = []
metadata = read_jsonl(metadata_path)
for sample in tqdm(metadata):
    image_path = os.path.join(os.path.dirname(metadata_path), sample["file_name"])
    mask_path = image_path.replace("/nodules_6mm/", "/masks_6mm/")
    assert os.path.exists(image_path) and os.path.exists(mask_path)
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path))
    if image.shape[0] >= CROP_SIZE:
        image, _, _ = crop_center(image, CROP_SIZE, CROP_SIZE)
        mask,  row_offset, col_offset = crop_center(mask, CROP_SIZE, CROP_SIZE)
        borders = mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum()
        if (len(np.unique(mask)) > 1) and borders == 0:
            bbx = np.fromstring(sample["bbx"], dtype=np.int16, sep=" ")
            zoom_factor = float(sample["pixel_spacing"])/0.6
            bbx[:4] = bbx[:4] * zoom_factor
            bbx = bbx.astype(np.int16)
            bbx[:2] -= row_offset
            bbx[2:4] -= col_offset
            mask = mask.astype(np.float32)
            mask[bbx[0]:bbx[1], bbx[2]:bbx[3]] *= 2
            mask = mask/2
            mask = np.clip(mask, 0, 255).astype(np.uint8)
            sample["bbx"] = array2string(bbx, precision=1)
            sample["0ID"] = count
            out_image_path = os.path.join(img_out_dir, os.path.basename(image_path))
            out_mask_path = os.path.join(mask_out_dir, os.path.basename(mask_path))
            Image.fromarray(image).save(out_image_path)
            Image.fromarray(mask).save(out_mask_path)
            metadata2.append(sample)
            count += 1
            
            # create masks2 - squared bounding boxes
            mask2, sq_sizes_ = get_squared_bounding_boxes(mask)
            out_mask_path2 = os.path.join(mask_out_dir2, os.path.basename(mask_path))
            Image.fromarray(mask2).save(out_mask_path2)
            
write_jsonl(metadata2_path, metadata2)
