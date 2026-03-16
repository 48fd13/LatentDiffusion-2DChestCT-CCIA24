#!/usr/bin/env python3

import argparse
import os
import numpy as np
import glob

from fid import FID # requires !pip install scipy==1.9.1

from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms

def find_mask_center(mask):
    # Find the coordinates of the non-zero elements
    y_coords, x_coords = torch.where(mask == 1)
    
    # Find the boundaries of the square
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

    # Calculate the central pixel coordinates
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    return 2*int(center_x//2), 2*int(center_y//2)

def read_image_dir_for_FID_computation_nodule_level(input_dir, masks_dir, resolution=256, n_images=None):
    img_tf = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    nodule_tf = transforms.Resize((32,32), interpolation=transforms.InterpolationMode.BILINEAR)
    special_cases = 0
    mask_tf = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST)

    img_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    images, valid_paths = [], []
    sample_idx = 0
    for p in tqdm(img_paths):
        img = np.array(Image.open(p))
        img = torch.from_numpy(img).permute(2, 0, 1).type(torch.float32) / 255.
        if img.shape[1] != resolution:
            img = img_tf(img)
            
        # extract nodule
        mask_p = masks_dir + f"/{os.path.basename(p)}"
        assert os.path.exists(mask_p)
        mask = np.array(Image.open(mask_p))
        mask[mask<200] = 0
        mask = torch.from_numpy(mask).type(torch.float32) / 255.
        if len(mask.shape) == 3:
            mask = mask.permute(2, 0, 1)
        else:
            mask = mask.unsqueeze(0)
        if mask.shape[1] != resolution:
            mask = mask_tf(mask)
        if len(torch.unique(mask)) < 2:
            continue
        center_col, center_row = find_mask_center(mask[0])
        s = 32
        min_row = max(center_row - s//2, 0)
        max_row = min(center_row + s//2, img.shape[1]-1)
        min_col = max(center_col - s//2, 0)
        max_col = min(center_col + s//2, img.shape[2]-1)
        assert min_row < max_row and min_col < max_col
        nodule = img[:, min_row:max_row , min_col:max_col]
        if not (nodule.shape[1] == s and nodule.shape[2] == s):
            nodule = nodule_tf(nodule)
            special_cases += 1
        assert nodule.shape[1] == s and nodule.shape[2] == s
        images.append(nodule)
        valid_paths.append(sample_idx)
        sample_idx += 1
    print(f"found {special_cases} special cases") # in these cases the nodule crop is less than 32x32

    images = torch.stack(images)
    # certain FID implementations require at least 2048 images
    #https://github.com/mseitzer/pytorch-fid/issues/13
    if (n_images is not None) and (images.shape[0] < n_images):
        diff = n_images - images.shape[0]
        images = torch.cat([images, images[:diff]], dim=0)
    return images, valid_paths

def read_image_dir_for_FID_computation(input_dir, resolution=256, n_images=None):
    img_tf = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    
    img_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    images = []
    for p in tqdm(img_paths):
        img = np.array(Image.open(p))
        img = torch.from_numpy(img).permute(2, 0, 1).type(torch.float32) / 255.
        if img.shape[1] != resolution:
            img = img_tf(img)
        images.append(img)
    images = torch.stack(images)
    # certain FID implementations require at least 2048 images
    #https://github.com/mseitzer/pytorch-fid/issues/13
    if (n_images is not None) and (images.shape[0] < n_images):
        diff = n_images - images.shape[0]
        images = torch.cat([images, images[:diff]], dim=0)
    return images




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate N synthetic images and save images/masks/metadata to disk.")
    p.add_argument(
        "--real_images_dir",
        required=True,
        help='Directory containing real images.',
    )
    p.add_argument(
        "--synthetic_images_dir",
        required=True,
        help='Directory containing synthetic images.',
    )
    p.add_argument(
        "--nodule_level",
        action="store_true",
        help='Additionally compute FID at nodule level, using nodule 32x32 crops.',
    )
    p.add_argument(
        "--real_masks_dir",
        default=None,
        help='Directory containing the nodule masks of real images for computing FID at nodule level.'
    )
    p.add_argument(
        "--synthetic_masks_dir",
        default=None,
        help='Directory containing the nodule masks of synthetic images for computing FID at nodule level.'
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    assert os.path.exists(args.real_images_dir)
    assert os.path.exists(args.synthetic_images_dir)

    print(f"\nReal images dir: {args.real_images_dir}")
    print(f"Synthetic images dir: {args.synthetic_images_dir}")

    print("\n-------------------------------------")
    print("Running FID...\n")

    # read real images
    real_images = read_image_dir_for_FID_computation(args.real_images_dir)
    print(f"Real images loaded. shape is {real_images.shape}")
    # read synthetic images
    fake_images = read_image_dir_for_FID_computation(args.synthetic_images_dir)
    print(f"Fake images loaded. shape is {fake_images.shape}")

    # compute FID
    fid = FID(real_images.cuda(), device="cuda")
    fid_score = fid.calculate_FID(fake_images.cuda())
    print(f"Output FID score: {fid_score:6.2f}")

    if args.nodule_level:

        del real_images
        del fake_images
        assert args.real_masks_dir is not None
        assert args.synthetic_masks_dir is not None
        assert os.path.exists(args.real_masks_dir)
        assert os.path.exists(args.synthetic_masks_dir)

        print("\n-------------------------------------")
        print("Now running FID at nodule level...\n")
        # read real images
        real_nodules, valid_real = read_image_dir_for_FID_computation_nodule_level(args.real_images_dir, args.real_masks_dir)
        print(f"Real images loaded. shape is {real_nodules.shape}")
        # read synthetic images
        fake_nodules, valid_fake = read_image_dir_for_FID_computation_nodule_level(args.synthetic_images_dir, args.synthetic_masks_dir)
        print(f"Fake images loaded. shape is {fake_nodules.shape}")

        # compute FID
        fid = FID(real_nodules.cuda(), device="cuda")
        fid_score = fid.calculate_FID(fake_nodules.cuda())
        print(f"Output FID score (nodule level): {fid_score:6.2f}")

    print("\nDONE")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())