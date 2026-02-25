import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel, AutoencoderKL, VQModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# @RRT
import sys
sys.path.insert(0, "src")

from pipeline import *
from utils_lidc import *
from custom_unet_cond import *

import datetime

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)



def main(args):

    logging_dir = args.logging_dir #os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Set the random seed manually for reproducibility.
    if args.acc_seed is not None:
        set_seed(args.acc_seed)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

                model_names = ["unet", "emb"]
                for i, (model, folder_name) in enumerate(zip(models, model_names)):
                    model.save_pretrained(os.path.join(output_dir, folder_name))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            model_names = ["unet", "emb"]
            for i, (model, folder_name) in enumerate(zip(models, model_names)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder=foder_name)
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load pretrained VAE model
    vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    #vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")


    # Freeze the VAE model
    vae.requires_grad_(False)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    # Initialize the model
    if args.model_config_name_or_path is None:
        #model = UNet2DModel(sample_size=args.resolution // vae_scale_factor, in_channels=3, out_channels=3)
        #model = UNet2DModel(sample_size=args.resolution // vae_scale_factor, in_channels=3+1*(3+args.emb_size), out_channels=3)
        #model = UNet2DModel(sample_size=args.resolution // vae_scale_factor, in_channels=3, out_channels=3, num_class_embeds=6*10)
        #model = UNet2DConditionModel(sample_size=args.resolution // vae_scale_factor, in_channels=3+1*(3+args.emb_size), out_channels=3)

        #Compvis: https://github.com/CompVis/stable-diffusion
        #model = Model(in_channels=3+3, out_ch=3, ch=128, num_res_blocks=2, 
        #              resolution=args.resolution // vae_scale_factor, attn_resolutions=[32, 16, 8])
        #Inverse SR: https://github.com/BioMedAI-UCSC/InverseSR
        
        mask_channels = 3 if args.encode_mask else 1

        if args.self_attention:
            model = UNetModel(image_size=(args.resolution // vae_scale_factor), in_channels=3 + mask_channels, out_channels=3,
                            model_channels=128, num_res_blocks=2, attention_resolutions=[32, 16, 8],
                            use_spatial_transformer=False)
        else:
            latent_dim = (args.resolution // vae_scale_factor)
            cond_vec_len = (latent_dim*latent_dim*mask_channels)
            in_channels = 3 + mask_channels
            if args.nodule_attributes:
                cond_vec_len += args.emb_size
                if args.nodule_attributes_v2:
                    in_channels += args.emb_size
            model = UNetModel(image_size=(args.resolution // vae_scale_factor), in_channels=in_channels, out_channels=3,
                                model_channels=128, num_res_blocks=2, attention_resolutions=[32, 16, 8],
                                use_spatial_transformer=True, context_dim=cond_vec_len)
        
        #model = UNetModel(image_size=(args.resolution // vae_scale_factor), in_channels=6, out_channels=3, model_channels=128, num_res_blocks=2, 
        #attention_resolutions=[])

    else:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)
        #model = UNet2DConditionModel.from_config(config)

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNetModel, #UNet2DModel,
            model_config=model.config_,
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # Initialitze the nodule features embedding
    metadata_path = os.path.join(args.dataset_name, "metadata.jsonl")
    assert os.path.exists(metadata_path)
    feature_labels = ["sphericity", "lobulation", "spiculation", "margin", "texture"]
    nodule_features_emb = NoduleFeaturesEmbedding(feature_labels, args.vocab_len, args.emb_size)
    nodule_features_emb.to(accelerator.device)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *nodule_features_emb.params],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    elif args.train_data_files is not None:
        dataset = load_dataset("imagefolder", data_files=args.train_data_files, split="train")
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB")) for image in examples["image"]]
        return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    #dataset.set_transform(transform_images)
    dataset.set_transform(parse_lidc_metadata)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, nodule_features_emb, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, nodule_features_emb, optimizer, train_dataloader, lr_scheduler
    )

    vae = vae.to(accelerator.device, dtype=weight_dtype)

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)


    # Train!
    masks_dir = args.dataset_name.replace("nodules", "masks") + "_sq"
    assert os.path.exists(masks_dir)
    mask_generator = RandomMaskGenerator(masks_dir)
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):

            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            clean_images = batch["image"].type(weight_dtype)
            masks = load_nodule_masks(batch["img_ids"], metadata_path, masks_dir, resolution=256, device=clean_images.device, bbxmask=False)
            masks = masks.type(weight_dtype)

            latents = vae.encode(clean_images).latents
            latents = latents * 0.18215 #vae.config.scaling_factor

            if args.encode_mask:
                mask_latents = vae.encode(masks).latents
                mask_latents = mask_latents * 0.18215
            else:
                # simply downsample the masks
                compressed_h, compressed_w = latents.shape[-2], latents.shape[-1]
                mask_t = transforms.Resize((compressed_h, compressed_w), interpolation=transforms.InterpolationMode.NEAREST)
                mask_latents = torch.cat([mask_t(m).unsqueeze(0) for m in masks])
                # use only one channel per binary mask instead of 3
                mask_latents = torch.cat([m[0].unsqueeze(0).unsqueeze(0) for m in mask_latents])
                masks = torch.cat([m[0].unsqueeze(0).unsqueeze(0) for m in masks])

            # Sample noise that we'll add to the images
            noise = torch.randn(latents.shape, dtype=weight_dtype, device=latents.device)
            bsz = latents.shape[0]  # batch size
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Add masks
            noisy_latents = torch.cat((noisy_latents, mask_latents), 1)
            cond_latents = mask_latents.view(bsz, -1)

            # inspired by https://github.com/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb
            if args.nodule_attributes:
                emb_vec = nodule_features_emb(batch) # nodule features embedding
                cond_latents = torch.cat([cond_latents, emb_vec], 1)

                if args.nodule_attributes_v2:
                    noisy_latents = merge_input_with_class_cond_embedding(noisy_latents, emb_vec)
                    # noisy latents shape is (batch_size, 3(rgb) + mask_channels + emb_dim, 64, 64)
                    # feature dim is 3 + 3 + 10 because latent dim is 3 for both image and mask, and len of attributes embedding vec is 10

            with accelerator.accumulate(model):
                # Predict the noise residual

                # UNet2DConditionModel with encoder_hidden_states
                # UNet2DModel with class_labels
                # UNet2DConditionModel with added_cond_kwargs
                #cross_attention_kwargs = {"mask": mask_latents}
                #model_output = model(sample=noisy_latents, timestep=timesteps, encoder_hidden_states=mask_latents).sample
                
                #COMPVIS model
                #model_output = model(x=noisy_latents, t=timesteps, context=mask_latents)
                #INVERSE SR model original context shape is (1, 1, 4), as they have batch size 1, 1 embedding per sample, and 4-len embedding (gender, age, ventricular, brain volume)
                model_output = model(x=noisy_latents, t=timesteps, context=cond_latents)


                if args.prediction_type == "epsilon": # this is the default loss that compares predicted vs ground truth noise
                    loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    # use SNR weighting from distillation paper
                    loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
                    loss = loss.mean()
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)
                nodule_features_emb = accelerator.unwrap_model(nodule_features_emb)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = CondLatentDiffusionPipeline_LIDC(
                    vae=vae,
                    unet=unet,
                    scheduler=noise_scheduler,
                    emb=nodule_features_emb,
                    mask_generator=mask_generator,
                    encode_mask=args.encode_mask,
                    nodule_attributes=args.nodule_attributes
                )

                generator = torch.Generator(device=vae.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                output = pipeline(
                    generator=generator,
                    height = 256,
                    width = 256,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="numpy",
                    return_dict=False
                )
                images, input_masks, output_masks, nodule_features = output
                images = merge_images_with_masks(images, input_masks)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)
                nodule_features_emb = accelerator.unwrap_model(nodule_features_emb)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                pipeline = CondLatentDiffusionPipeline_LIDC(
                    vae=vae,
                    unet=unet,
                    scheduler=noise_scheduler,
                    emb=nodule_features_emb,
                    mask_generator=mask_generator,
                    encode_mask=args.encode_mask,
                    nodule_attributes=args.nodule_attributes
                )

                params_dict = {"unet": unet.parameters(), "emb": nodule_features_emb.params}
                pipeline.save_pretrained(args.output_dir, params=params_dict)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=args.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
