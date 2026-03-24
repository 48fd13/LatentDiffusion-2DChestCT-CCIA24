#!/usr/bin/env python3

import argparse
import os
import sys
import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Keep original behavior of importing project utilities.
sys.path.insert(1, "./src")
from utils_lidc import *  # noqa: F401,F403
from custom_unet_cond import *
from utils_lidc import *

def save_pipeline_output_to_disk(output, batch_idx, batch_size, out_dir, save_masks):
    metadata_path = out_dir + "/ims/metadata.jsonl"
    os.makedirs(out_dir + "/images", exist_ok=True)
    if save_masks:
        # conditional synthesis
        images, input_masks, output_masks, nodule_features = output
        super_images = merge_images_with_masks(images, input_masks)
        os.makedirs(out_dir + "/masks", exist_ok=True)
        os.makedirs(out_dir + "/overlay", exist_ok=True)
        if nodule_features is not None:
            for k in nodule_features.keys():
                nodule_features[k] = nodule_features[k].detach().cpu().numpy()
    else:
        # unconditional synthesis
        images = output[0]

    metadata = []
    for idx in range(min(batch_size, images.shape[0])):
        local_d = {}
        global_img_idx = batch_idx * batch_size + idx
        local_d["0ID"] = int(global_img_idx)
        local_d["file_name"] = f"{global_img_idx:05d}.png"

        img = Image.fromarray((images[idx] * 255).astype(np.uint8))
        img.save(out_dir + f"/images/{global_img_idx:05d}.png")

        if save_masks:
            mask = np.dstack([input_masks[idx], input_masks[idx], input_masks[idx]])
            img = Image.fromarray((mask * 255).astype(np.uint8))
            img.save(out_dir + f"/masks/{global_img_idx:05d}.png")

            img = Image.fromarray((super_images[idx] * 255).astype(np.uint8))
            img.save(out_dir + f"/overlay/{global_img_idx:05d}.png")

            local_d["area"] = int(np.sum(input_masks[idx, :, :, 0].flatten() > 0))
            if nodule_features is not None:
                for k in nodule_features:
                    local_d[k] = int(nodule_features[k][idx])
        metadata.append(local_d)

    if os.path.exists(metadata_path):
        update_jsonl(metadata, metadata_path)
    else:
        write_jsonl(metadata_path, metadata)


def generate_N_images(pipeline, out_dir: str, n_images: int, batch_size: int = 16, inference_steps: int | None = None, seed: int = 0):
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    save_masks = False if pipeline.mask_generator is None else True

    if inference_steps is None:
        inference_steps = pipeline.scheduler.config.num_train_timesteps

    im_size = pipeline.vae.sample_size

    for batch_idx in tqdm(range(n_images // batch_size)):
        output = pipeline(
            height=im_size,
            width=im_size,
            generator=generator,
            batch_size=batch_size,
            num_inference_steps=inference_steps,
            output_type="numpy",
            return_dict=False,
        )
        save_pipeline_output_to_disk(output, batch_idx, batch_size, out_dir, save_masks=save_masks)

    if n_images % batch_size > 0:
        batch_idx = n_images // batch_size
        output = pipeline(
            height=im_size,
            width=im_size,
            generator=generator,
            batch_size=n_images % batch_size,
            num_inference_steps=inference_steps,
            output_type="numpy",
            return_dict=False,
        )        
        save_pipeline_output_to_disk(output, batch_idx + 1, batch_size, out_dir, save_masks=save_masks)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate N synthetic images and save images/masks/metadata to disk.")

    # Keep original defaults, but make them configurable.
    p.add_argument(
        "--ckpt_dir",
        required=True,
        help='Checkpoint path (e.g. data/ckpts/2024-06-12_18-01-52_FINAL_CCIA24_unconditional_latent_BS8)',
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help='Output directory (e.g. data/outputs).',
    )
    p.add_argument(
        "--masks_dir",
        default=None,
        help="Dataset path used by load_pipeline (e.g. data/train_data/nodules_6mm_512x512)",
    )
    p.add_argument("--n_images", type=int, default=4, help="Number of images to generate (default: 2048).")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 16).")
    p.add_argument(
        "--inference_steps",
        type=int,
        default=None,
        help="Number of diffusion inference steps. Default: scheduler.config.num_train_timesteps.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for torch Generator (default: 0).")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, generate even if out_dir already exists (files may be appended/overwritten).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    assert os.path.exists(args.ckpt_dir)

    pipeline = load_pipeline(args.ckpt_dir, args.masks_dir)

    if os.path.exists(args.out_dir) and not args.overwrite:
        print(f"out_dir {args.out_dir} already exists (use --overwrite to force)")
        print("DONE")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    generate_N_images(
        pipeline=pipeline,
        out_dir=args.out_dir,
        n_images=args.n_images,
        batch_size=args.batch_size,
        inference_steps=args.inference_steps,
        seed=args.seed,
    )

    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
