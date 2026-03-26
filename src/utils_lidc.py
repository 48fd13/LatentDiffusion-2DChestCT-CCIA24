import argparse
import datetime
import numpy as np
import os
import torch
from torch import nn
from torchvision import transforms
import json
from PIL import Image
import random
from diffusers import ModelMixin, ConfigMixin
import scipy.ndimage
import glob

from pipeline import *
import json
import diffusers
from custom_unet_cond import *
from math import ceil


# create a custom module that can be plugged into a diffusion pipeline from the diffusers library
# this was helpful https://github.com/huggingface/diffusers/issues/3231
# when using diffusers new modules cannot inheritate from torch.nn.Module and be saved with save_pretrained
# instead, new modules have to inheritate from ModelMixin, ConfigMixin



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddpm-model-256",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--acc_seed",
        type=int,
        default=None,
        help="A seed to reproduce the training. If not set, the seed will be random.",
    )
    parser.add_argument(
        "--train_data_files",
        type=str,
        default=None,
        help=(
            "The files of the training data. The files must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument("--emb_size", type=int, default=10, help="Dimensionality of nodule attributes embedding.")
    parser.add_argument("--vocab_len", type=int, default=6, help="Number of nodule attributes.")
    parser.add_argument("--masks_dir", type=str, default=None, help="Directory where nodule masks are located.")
    parser.add_argument("--encode_mask", action="store_true", help="Whether or not to encode the localization mask.")
    parser.add_argument("--self_attention", action="store_true", help="Use self-attention.")
    parser.add_argument("--nodule_attributes", action="store_true", help="Use nodule attribute embeddings")


    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_files is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    exp_id = args.output_dir.split("/")[-1]
    print(f"\n\nRunning exp {exp_id}\n\n")

    return args



class NoduleFeaturesEmbedding(ModelMixin, ConfigMixin):

    def __init__(self, feature_labels, vocab_len, emb_size):
        super(NoduleFeaturesEmbedding, self).__init__()

        self.vocab_len = vocab_len
        self.emb_size = emb_size
        self.feature_labels = feature_labels
        self.n_emb = len(feature_labels)

        for label in self.feature_labels:
            setattr(self, f"emb_{label}", nn.Embedding(self.vocab_len, self.emb_size))

        in_features = self.n_emb * self.emb_size
        out_features = self.emb_size
        inner_features = out_features * 2 # 128 

        self.merging_block = torch.nn.Sequential(torch.nn.Linear(in_features, inner_features), nn.ReLU(),
                                                   torch.nn.Linear(inner_features, out_features), nn.ReLU())

        self.params = [*self.merging_block.parameters()]
        for label in self.feature_labels:
            self.params += [*getattr(self, f"emb_{label}").parameters()]

    def forward(self, x):
        
        out_vecs = []
        for label in self.feature_labels:
            #x[label].shape = batch_size
            #emb_vec.shape = (batch_size, emb_size)
            emb_vec = getattr(self, f"emb_{label}")(x[label]) 
            out_vecs.append(emb_vec)
        out_vecs = torch.stack(out_vecs, dim=1)
        batch_size = out_vecs.shape[0]
        #out_vecs.shape = (batch_size, n_emb, emb_size)
        #n_emb can be seen as the number of words in a sentence describing the nodule
        #emb_size is the dimensionality of the embedding vectors
        out = self.merging_block(out_vecs.view(batch_size, -1))
        return out

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save the configuration instance's parameters to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file to save a configuration instance's parameters.
        """
        config = {"_class_name": "NoduleFeaturesEmbedding", "_diffusers_version": diffusers.__version__,
            "feature_labels": self.feature_labels, "vocab_len": self.vocab_len, "emb_size": self.emb_size} 
        with open(json_file_path, "w", encoding="utf-8") as writer:
            json.dump(config, writer, indent=2)


def get_contours_mask(binary_mask, contour_width=2):


    def get_neighbors(distance):
        neighbors = []
        for i in range(-distance, distance + 1):
            for j in range(-distance, distance + 1):
                if (i, j) != (0, 0):
                    neighbors.append((i, j))
        return neighbors


    # idea: could be improved by iterating only in the white pixels
    if len(binary_mask.shape) > 2:
        binary_mask = binary_mask[:, :, 0]

    # Initialize the contour image
    binary_mask = binary_mask > 0
    contour_image = np.zeros_like(binary_mask)

    # Get the dimensions of the image
    rows, cols = binary_mask.shape

    # Iterate through each pixel in the binary mask
    for i in range(contour_width, rows - contour_width):
        for j in range(contour_width, cols - contour_width):
            # Check if the current pixel is part of the white square
            if binary_mask[i, j] == 1:
                # Check the neighbors at the given contour width distance
                for distance in range(1, contour_width + 1):
                    neighbors = get_neighbors(distance)
                    for di, dj in neighbors:
                        ni, nj = i + di, j + dj
                        if binary_mask[ni, nj] == 0:
                            # If any neighbor is black, mark this pixel as part of the contour
                            contour_image[i, j] = 1
                            break

    return contour_image[:, :, np.newaxis].astype(bool)

def merge_images_with_masks(images, masks):
    super_images = images.copy()
    super_mask = np.zeros_like(super_images).astype(bool)
    for idx in range(images.shape[0]):
        #mask = masks[idx].permute(1, 2, 0).detach().cpu().numpy().astype(bool)
        mask_ = masks[idx].astype(bool)
        #mask = mask_
        mask = get_contours_mask(mask_, contour_width=2)
        assert len(mask.shape) == 3
        if mask.shape[-1] == 1:
            mask = np.dstack([mask, mask, mask]) 
        mask[:, :, 1:] = False
        super_mask[idx] = mask
    super_images[super_mask] += 0.5
    super_images = np.clip(super_images, 0, 1)
    return super_images

def get_parameters(models):
    """
    Get all model parameters recursively
    models can be a list, a dictionary or a single pytorch model
    """
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:
        # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters

def generate_random_nodule_features(batch_size, labels, device):
    # valid values are 1, 2, 3, 4, 5
    d = {}
    for l in labels:
        x = np.random.randint(low=1, high=6, size=(batch_size,))
        d[l] = torch.from_numpy(x).to(device)
    return d

def read_jsonl(metadata_path):
    # returns a list of dictionaries
    with open(metadata_path, 'r') as f: 
        metadata = list(f)
    metadata = [json.loads(s) for s in metadata]
    return metadata

def write_jsonl(metadata_path, metadata):
    with open(metadata_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")

def update_jsonl(metadata, metadata_path):
    if os.path.exists(metadata_path):
        metadata_ = read_jsonl(metadata_path)
        metadata = metadata + metadata_
        unique_tuples = set(tuple(sorted(d.items())) for d in metadata)
        unique_dicts = [dict(t) for t in unique_tuples]
        metadata = sorted(unique_dicts, key=lambda x: x['file_name'])
        os.remove(metadata_path)
    write_jsonl(metadata_path, metadata)

def array2string(np_array, precision=2):
    s = np.array2string(np_array, precision=precision, separator=" ")
    s = s.replace("[", "").replace("]", "")
    return s

def string2array(s, dtype=np.float16):
    np_array = np.fromstring(s, dtype=dtype, sep=" ")
    return np_array

def get_first_layer_from_PIL_image(pil_image):
    x = Image.fromarray(np.array(pil_image)[:, :, 0])
    return x

def parse_lidc_metadata(batch, resolution=256):
    
    #augmentations = transforms.Compose([transforms.PILToTensor()])
    augmentations = transforms.Compose(
        [
            transforms.CenterCrop((512, 512)),
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    batch_size = len(batch['image'])

    out = {'image': [], 'img_ids': [], \
            'sphericity': [], 'margin': [], 'lobulation': [], 'spiculation': [], 'texture': [], \
            'area': [], 'pixel_spacing': [], 'slice_spacing': [], 'bbx': []}
    for i in range(batch_size):
        out['img_ids'].append(torch.tensor(batch['0ID'][i]).long())
        img = batch['image'][i].convert("RGB")
        #img = get_first_layer_from_PIL_image(img)
        t_img = augmentations(img)
        out['image'].append(t_img)
        parse_float = lambda x: random.choice(string2array(x, dtype=np.float32))
        for l in ['sphericity', 'margin', 'lobulation', 'spiculation', 'texture']:
            out[l].append(torch.tensor(parse_float(batch[l][i])).long())
        out['area'].append(torch.tensor(int(batch['area'][i])))
        out['pixel_spacing'].append(torch.tensor(float(batch['pixel_spacing'][i])))
        out['slice_spacing'].append(torch.tensor(float(batch['slice_spacing'][i])))
        out['bbx'].append(torch.tensor(string2array(batch['bbx'][i], dtype=np.int16)))
    return out


def transform_nodule_location_mask(mask, resolution):
    # mask in list_of_masks is a 512x512 numpy array

    augmentations = transforms.Compose(
        [
            transforms.CenterCrop((512, 512)),
            #transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
    mask = np.dstack([mask, mask, mask])
    t_mask = augmentations(Image.fromarray(mask))
    #t_mask = t_mask > 0.5
    return t_mask


def load_nodule_masks(image_ids, metadata_path, masks_dir, resolution, device="gpu", bbxmask=True):
    metadata = read_jsonl(metadata_path)
    masks = []
    for idx in image_ids:
        mask_path = os.path.join(masks_dir, metadata[idx]['file_name'])
        assert os.path.exists(mask_path) 
        bbx = np.fromstring(metadata[idx]['bbx'], dtype=np.int16, sep=" ")
        first_row, last_row, first_col, last_col  = bbx[0], bbx[1], bbx[2], bbx[3]
        mask = np.array(Image.open(mask_path))
        mask[mask<200] = 0
        if bbxmask:
            mask[first_row:last_row, first_col:last_col] = 255
        t_mask = transform_nodule_location_mask(mask, resolution)
        masks.append(t_mask)
    masks = torch.stack(masks).to(device)
    return masks

class RandomMaskGenerator():

    def __init__(self, masks_dir, verbose=True):
        assert os.path.exists(masks_dir)
        masks_paths = glob.glob(masks_dir + "/*.png")
        assert len(masks_paths) > 0

        if verbose:
            print("building random mask generator...")
        size = np.array(Image.open(masks_paths[0])).shape[0]
        self.mask_shape = (size, size)
        prob_counter = np.zeros(self.mask_shape)
        self.max_bbx_len = 80
        size_prob_counter = np.zeros(self.max_bbx_len + 1)
        for p in masks_paths:
            # nodule localization probabilities
            mask = np.array(Image.open(p)).astype(np.float32)
            if len(mask.shape) > 2: # this should not happen, masks are supposed to be binary png format
                print("warning: input masks are not grayscale, they have 3 channels (likely RGB)")
                mask = mask[:, :, 0]
            prob_counter += mask/255
            
            # nodule size probabilities
            binary_mask = (mask > 0).astype(np.uint8)
            labeled_mask, num_features = scipy.ndimage.label(binary_mask, structure=np.ones((3,3)))
            _, counts = np.unique(labeled_mask, return_counts=True)
            for nodule_bbx_sq_size in np.sqrt(counts[1:]):
                size_prob_counter[int(nodule_bbx_sq_size)] += 1

        self.prob_counter = prob_counter/prob_counter.max()
        self.current_mask_prob_counter = self.prob_counter.copy()
        self.size_prob_counter = size_prob_counter/size_prob_counter.max()
        if verbose:
            print("random mask generator is ready")

    def get_topleft_coordinates(self):
        h, w = self.current_mask_prob_counter.shape
        flat_image = self.current_mask_prob_counter.flatten()
        probabilities = flat_image / flat_image.sum()
        chosen_index = np.random.choice(len(flat_image), p=probabilities)
        row, col = np.unravel_index(chosen_index, (h, w))
        return row, col
        
    def insert_square(self, mask, sq_size, val):
        topleft_y, topleft_x = self.get_topleft_coordinates()
        topleft_x = min(topleft_x, self.mask_shape[1] - sq_size - 5)
        topleft_y = min(topleft_y, self.mask_shape[0] - sq_size - 5)
        mask[topleft_y:topleft_y + sq_size, topleft_x:topleft_x + sq_size] = np.ones((sq_size, sq_size)) * val
        self.current_mask_prob_counter[topleft_y:topleft_y + sq_size, topleft_x:topleft_x + sq_size] = 0
        return mask

    def generate_random_mask(self, sq_sizes=None):


        if sq_sizes is None:
            n_nodules = 1 #random.randint(1, 3) # max 3 nodules in a CT slice
            #sq_sizes = [random.randint(2, 80) for i in range(n_nodules)]
            min_size = 12
            probabilities = self.size_prob_counter[min_size:] / self.size_prob_counter[min_size:].sum()
            sq_sizes = [np.random.choice(np.arange(min_size, self.max_bbx_len + 1), p=probabilities) for i in range(n_nodules)]
            sq_sizes = [sq//2*2 for sq in sq_sizes]
        
        self.current_mask_prob_counter = self.prob_counter.copy()
        mask = np.zeros(self.mask_shape, dtype=np.uint8) # init mask
        mask = self.insert_square(mask, sq_sizes[0], 255) # insert first nodule
        if n_nodules > 1:
            for s in sq_sizes[1:]: # insert additional nodules (optional)
                mask = self.insert_square(mask, s, 127)
    
        # verify no overlap between bounding boxes
        binary_mask = (mask > 0).astype(np.uint8)
        labeled_mask, num_cc = scipy.ndimage.label(binary_mask, structure=np.ones((3,3)))
        
        if num_cc < len(sq_sizes):
            mask = self.generate_random_mask(sq_sizes)

        return mask

    def generate_n_masks_as_tensor(self, n_masks, resolution=256):
        masks = [self.generate_random_mask() for i in range(n_masks)]
        masks = [transform_nodule_location_mask(mask, resolution=resolution).unsqueeze(0) for mask in masks]
        masks = torch.cat(masks, dim=0) # shape is (n_masks, 3, 256, 256)
        return masks


class CondLatentDiffusionPipeline_LIDC(LatentDiffusionPipelineBase):
    def __init__(
            self,
            vae: VQModel,
            scheduler: Union[
                DDIMScheduler,
                DDPMScheduler,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
                LMSDiscreteScheduler,
                PNDMScheduler,
            ],
            unet: UNet2DModel,
            emb: NoduleFeaturesEmbedding,
            mask_generator: RandomMaskGenerator,
            encode_mask: Optional[bool] = False,
            nodule_attributes: Optional[bool] = False
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            emb=emb,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.mask_generator = mask_generator
        self.encode_mask = encode_mask
        self.nodule_attributes = nodule_attributes

    @torch.no_grad()
    def __call__(
            self,
            batch_size: int = 1,  # default to generate a single image
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: Optional[int] = 50,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            nodule_features: Optional[dict] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            eta: Optional[float] = 0.0,
            **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        device, dtype = self.vae.device, self.vae.dtype
        
        #parepare latents
        latents = self.prepare_latents(batch_size, 3, height, width,
                                       dtype, device, generator, latents)
        # latents.shape = (batch_size, 3, latent_height, latent_width)

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # init scheduler
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.vae.device).long()
        noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device)
        _ = self.scheduler.add_noise(latents, noise, timesteps)


        unconditional = self.mask_generator is None

        # prepare conditional data
        if not unconditional:
            # (optional) load nodule attributes
            # nodule_features is expected to be a dict. E.g. {'sphericity': 4, 'margin': 1, 'lobulation': 2, 'texture': 5, 'spiculation': 1}
            if self.nodule_attributes:
                assert self.emb is not None
                if nodule_features is None:
                    nodule_features = generate_random_nodule_features(batch_size, self.emb.feature_labels, self.emb.device)
                emb_vec = self.emb(nodule_features)
            else:
                nodule_features = None

            # load masks
            masks = self.mask_generator.generate_n_masks_as_tensor(batch_size, resolution=height).to(device).type(self.vae.dtype)
            #masks = masks/255.
            # compress masks
            if self.encode_mask:
                mask_latents = self.vae.encode(masks).latents * 0.18215
                # mask_latents.shape = (batch_size, 3, latent_height, latent_width)
            else:
                # simply downsample the masks
                compressed_h, compressed_w = height // self.vae_scale_factor, width // self.vae_scale_factor
                mask_t = transforms.Resize((compressed_h, compressed_w), interpolation=transforms.InterpolationMode.NEAREST)
                mask_latents = torch.cat([mask_t(m).unsqueeze(0) for m in masks])
                # use only one channel per binary mask instead of 3
                mask_latents = torch.cat([m[0].unsqueeze(0).unsqueeze(0) for m in mask_latents])
                masks = torch.cat([m[0].unsqueeze(0).unsqueeze(0) for m in masks])
                # mask_latents.shape = (batch_size, 1, latent_height, latent_width)

        # run denoising iterative process
        timesteps_ = list(self.scheduler.timesteps)
        total_steps = len(timesteps_)
        update_every = max(1, ceil(total_steps / 5))
        with self.progress_bar(total=total_steps) as pbar:
            done = 0
            shown = 0

            for t in timesteps_:

                timesteps = t.repeat((batch_size,)).to(latents.device).long()

                latents = self.scheduler.scale_model_input(latents, t)

                if unconditional:
                    noisy_latents = latents
                    noise_pred = self.unet(noisy_latents, timesteps).sample
                else:
                    noisy_latents = torch.cat((latents, mask_latents), 1)
                    cond_latents = mask_latents.view(batch_size, -1)
                    if self.nodule_attributes:
                        cond_latents = torch.cat([cond_latents, emb_vec], 1)
                    noise_pred = self.unet(x=noisy_latents, t=timesteps, context=cond_latents)

                # compute the previous noisy sample x_t -> x_t-1
                latents = step(self.scheduler,
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                done += 1
                if done % update_every == 0 or done == total_steps:
                    pbar.update(done - shown)
                    shown = done

        # scale and decode the image latents with vae
        images = self.decode_latents(latents) # shape is (B, H, W, 3)
        
        # force grayscale intensities
        images = np.mean(images, axis=-1, keepdims=True)
        images = np.repeat(images, 3, axis=-1)

        if output_type == "pil":
            images = self.numpy_to_pil(images)

        if unconditional:
            if not return_dict:
                return (images,)
        else:
            input_masks = masks.permute(0, 2, 3, 1).detach().cpu().numpy()
            if self.encode_mask:
                output_masks = self.decode_latents(mask_latents)
            else:
                mask_t = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST)
                output_masks = torch.cat([mask_t(m)[0].unsqueeze(0).unsqueeze(0) for m in mask_latents])
                output_masks = output_masks.permute(0, 2, 3, 1).detach().cpu().numpy()
            #input_masks *= 255
            #output_masks *=255
            if not return_dict:
                return (images, input_masks, output_masks, nodule_features)

        return ImagePipelineOutput(images=images)


from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput

def step(
        self: DDPMScheduler,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise
        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

def load_pipeline(ckpt_dir, masks_dir=None, verbose=True, device="cuda"):

    if verbose:
        print("Loading Diffusion pipeline from:")
        print(f"    - {ckpt_dir}\n")

    # load vae
    vae = VQModel.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")
    vae.requires_grad_(False)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae.to(device)
    if verbose:
        print("VQ-VAE loaded")

    # load u-net
    unet_config_json_path = ckpt_dir + "/unet/config.json"
    assert os.path.exists(unet_config_json_path)
    with open(unet_config_json_path) as f:
        unet_config_dict = json.load(f)
        if unet_config_dict["_class_name"] == "UNetModel":
            unet = UNetModel.from_pretrained(ckpt_dir, subfolder=f"unet")
        else:
            unet = UNet2DModel.from_pretrained(ckpt_dir, subfolder=f"unet")
    unet.to(device)
    if verbose:
        print("U-Net model loaded")
    
    # (optional) load nodule features embedding module
    emb = None
    if os.path.exists(ckpt_dir + "/emb"):
        emb = NoduleFeaturesEmbedding.from_pretrained(ckpt_dir, subfolder="emb")
        emb.to(device)
        if verbose:
            print("Nodule Features Embedding loaded")

    # (optional) load heuristic mask generator for conditional synthesis
    # helps control the size and location of output pulmonary nodules
    mask_generator = None
    if masks_dir is not None:
        assert os.path.exists(masks_dir)
        print("Creating synthetic mask generator...")
        mask_generator = RandomMaskGenerator(masks_dir, verbose=False)
        if verbose:
            print("...done!")

    scheduler_config_path = ckpt_dir + "/scheduler/scheduler_config.json" 
    noise_scheduler = DDPMScheduler.from_config(scheduler_config_path)

    pipeline = CondLatentDiffusionPipeline_LIDC(
        vae=vae,
        unet=unet,
        scheduler=noise_scheduler,
        emb=emb,
        mask_generator=mask_generator,
        nodule_attributes=False if emb is None else True)
    print("Diffusion pipeline is ready\n")
    
    return pipeline
