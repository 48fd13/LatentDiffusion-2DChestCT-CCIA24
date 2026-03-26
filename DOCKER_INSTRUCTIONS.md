# Docker usage guide

This document explains how to build and run the Docker image for **LatentDiffusion-2DChestCT-CCIA24**.

It assumes that a compatible `Dockerfile` is already present in the root of the repository.

---

## 1. What this Docker image does

The image packages the project code and Python environment so the repository can be run consistently on a server without creating a separate virtual environment.

The image is intended for GPU-based execution.

Typical uses:
- image generation / inference
- model training
- FID evaluation

The dataset and checkpoints should **not** be copied into the image. They should remain outside the image and be mounted at runtime.

---

## 2. Prerequisites on the server

Before running the container, the host machine must satisfy all of the following:

### Docker installed and running
Check:

```bash
docker --version
sudo systemctl status docker
```

If Docker is installed but not started:

```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### Permission to use Docker
If `docker build` or `docker run` fails with a message like:

```bash
permission denied while trying to connect to the Docker daemon socket
```

then either use `sudo` temporarily:

```bash
sudo docker build -t latentdiffusion-2dchestct-ccia24:latest .
```

or add your user to the `docker` group:

```bash
sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker $USER
newgrp docker
```

Then test:

```bash
docker ps
```

### NVIDIA GPU drivers installed
Check:

```bash
nvidia-smi
```

This must show the available GPUs.

### NVIDIA Container Toolkit installed
If this command fails:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

with an error like:

```bash
could not select device driver "" with capabilities: [[gpu]]
```

then Docker does not yet have GPU support configured.

Install and configure NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then validate again:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

---

## 3. Repository layout expected at runtime

The commands below assume you are in the root of the repository and that your data lives in a local `data/` folder.

Typical expected layout:

```text
LatentDiffusion-2DChestCT-CCIA24/
├── Dockerfile
├── requirements.txt
├── train_lidc.py
├── generate_N_images.py
├── compute_fid.py
└── data/
    ├── ckpts/
    ├── outputs/
    └── train_data/
```

Inside the container, this local folder will be mounted to:

```text
/opt/app/data
```

---

## 4. Build the Docker image

From the repository root:

```bash
docker build -t latentdiffusion-2dchestct-ccia24:latest .
```

If your server requires `sudo` for Docker:

```bash
sudo docker build -t latentdiffusion-2dchestct-ccia24:latest .
```

---

## 5. Quick validation tests

### Check that the image exists

```bash
docker images | grep latentdiffusion-2dchestct-ccia24
```

### Check Python and PyTorch inside the container

```bash
docker run --rm latentdiffusion-2dchestct-ccia24:latest -c "import torch; print(torch.__version__)"
```

### Check GPU visibility inside the project container

```bash
docker run --rm --gpus all \
  latentdiffusion-2dchestct-ccia24:latest \
  -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Expected result:
- `True`
- number of visible GPUs greater than or equal to 1

---

## 6. Run inference

Mount the host `data/` directory into the container and run the generation script.

### Unconditional generation

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/opt/app/data \
  latentdiffusion-2dchestct-ccia24:latest \
  generate_N_images.py \
    --ckpt_dir /opt/app/data/ckpts/2024-06-12_18-01-52_FINAL_CCIA24_unconditional_latent_BS8 \
    --out_dir /opt/app/data/outputs/test_run \
    --n_images 4 \
    --batch_size 4 \
    --overwrite
```

### Conditional generation

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/opt/app/data \
  latentdiffusion-2dchestct-ccia24:latest \
  generate_N_images.py \
    --ckpt_dir /opt/app/data/ckpts/2024-07-12_19-57-59_FINAL_CCIA24_model3_crossattention_locmask_noduleattributes_maskv2 \
    --out_dir /opt/app/data/outputs/test_conditional \
    --masks_dir /opt/app/data/train_data/masks_6mm_512x512_sq \
    --n_images 4 \
    --batch_size 4 \
    --overwrite
```

Notes:
- `--ckpt_dir` must point to a valid checkpoint directory mounted under `/opt/app/data/ckpts`
- output files will be written to the mounted host directory under `data/outputs`

---

## 7. Run training

Because the `Dockerfile` uses:

```dockerfile
ENTRYPOINT ["python3"]
```

training is easiest by opening an interactive shell in the container and launching `accelerate` from there.

Start a shell:

```bash
docker run --rm -it --gpus all \
  --entrypoint bash \
  -v $(pwd)/data:/opt/app/data \
  latentdiffusion-2dchestct-ccia24:latest
```

Then, inside the container, run:

```bash
accelerate launch train_lidc.py \
  --dataset_name=/opt/app/data/train_data/nodules_6mm_512x512 \
  --masks_dir=/opt/app/data/train_data/masks_6mm_512x512_sq \
  --resolution=256 \
  --output_dir=/opt/app/data/logs/my_run \
  --logging_dir=/opt/app/data/logs/my_run \
  --train_batch_size=16 \
  --num_epochs=305 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --checkpointing_steps=5000 \
  --save_images_epochs=20 \
  --save_model_epochs=20 \
  --ddpm_num_inference_steps=1000 \
  --mixed_precision=no
```

### Optional training flags for large jobs

For memory-heavy runs, these Docker options may help:

```bash
--shm-size=16g
```

Example:

```bash
docker run --rm -it --gpus all \
  --shm-size=16g \
  --entrypoint bash \
  -v $(pwd)/data:/opt/app/data \
  latentdiffusion-2dchestct-ccia24:latest
```

---

## 8. Run FID evaluation

`compute_fid.py` uses CUDA directly, so GPU access is required.

```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/opt/app/data \
  latentdiffusion-2dchestct-ccia24:latest \
  compute_fid.py \
    --real_images_dir /opt/app/data/train_data/nodules_6mm_512x512 \
    --synthetic_images_dir /opt/app/data/outputs/test_run/ims
```

---

## 9. Useful Docker commands

### Open an interactive shell in the image

```bash
docker run --rm -it --gpus all \
  --entrypoint bash \
  -v $(pwd)/data:/opt/app/data \
  latentdiffusion-2dchestct-ccia24:latest
```

### List Docker images

```bash
docker images
```

### Remove the image

```bash
docker rmi latentdiffusion-2dchestct-ccia24:latest
```

### Show container logs for a non-interactive run

If you run a container without `--rm`, first find the container id:

```bash
docker ps -a
```

Then inspect logs:

```bash
docker logs <container_id>
```

---

## 10. Common problems and fixes

### Problem: permission denied while connecting to Docker socket

Example error:

```bash
permission denied while trying to connect to the Docker daemon socket
```

Fix:
- use `sudo docker ...`, or
- add your user to the `docker` group

Commands:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

### Problem: GPU error with `--gpus all`

Example error:

```bash
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

Cause:
- NVIDIA Container Toolkit is not installed or not configured

Fix:
- install `nvidia-container-toolkit`
- run `sudo nvidia-ctk runtime configure --runtime=docker`
- restart Docker

Validation:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

---

### Problem: data or checkpoints not found

Cause:
- the host `data/` folder is not mounted
- the path inside the command is wrong

Fix:
- ensure the run command contains:

```bash
-v $(pwd)/data:/opt/app/data
```

- ensure the checkpoint directory exists on the host, for example:

```bash
ls $(pwd)/data/ckpts
```

---

### Problem: output files are missing on the host

Cause:
- output path inside the container is not under `/opt/app/data`

Fix:
- write outputs only inside the mounted path, for example:

```bash
--out_dir /opt/app/data/outputs/test_run
```

---

## 11. Notes about the Dockerfile

The provided `Dockerfile` is GPU-oriented and uses:
- `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- Python 3.10
- PyTorch 2.5.1 with CUDA 12.1

It also includes:

```dockerfile
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

These environment variables are appropriate for GPU execution, but they do **not** replace the need to configure the host with NVIDIA Container Toolkit.

---

## 12. Recommended workflow

A practical sequence for first-time use is:

```bash
# 1. verify host GPU
nvidia-smi

# 2. verify Docker
sudo systemctl status docker

# 3. verify Docker GPU support
docker run --rm --gpus all ubuntu nvidia-smi

# 4. build the image
docker build -t latentdiffusion-2dchestct-ccia24:latest .

# 5. test torch CUDA inside the image
docker run --rm --gpus all \
  latentdiffusion-2dchestct-ccia24:latest \
  -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# 6. run inference
docker run --rm --gpus all \
  -v $(pwd)/data:/opt/app/data \
  latentdiffusion-2dchestct-ccia24:latest \
  generate_N_images.py \
    --ckpt_dir /opt/app/data/ckpts/2024-06-12_18-01-52_FINAL_CCIA24_unconditional_latent_BS8 \
    --out_dir /opt/app/data/outputs/test_run \
    --n_images 4 \
    --batch_size 4 \
    --overwrite
```

---

## 13. Export docker image

```text
 docker save latentdiffusion-2dchestct-ccia24:latest | gzip > latentdiffusion-2dchestct-ccia24-1.0.tar.gz
```
That output tar.gz will contain:

- the Docker image layers
- metadata needed to load the image again

It will not contain:

- generated outputs stored in `./data/outputs`


