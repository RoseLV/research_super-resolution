# precipitation-diffusion-downscaling 
This repository implements a diffusion-based model for precipitation downscaling. This is the codebase for [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

# Usage
This section of the README walks through how to train and sample from a model.

## Installation
Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on. 

## Train

```
bash train.sh
```
### Training Parameters

#### `MODEL_FLAGS`
Specifies the model architecture:

- `--num_channels`: Number of channels in the model (default: **128**).
- `--num_res_blocks`: Number of residual blocks (default: **2**).
- `--learn_sigma`: Enables sigma learning (default: **True**).
- `--class_cond`: Class-conditional model (default: **False**).
- `--large_size` and `--small_size`: Defines the size of the large and small input images (default: **128** and **16**).

#### `DIFFUSION_FLAGS`
Configures the diffusion process:

- `--diffusion_steps`: Number of diffusion steps (default: **4000**).
- `--noise_schedule`: Noise schedule type (default: **linear**).
- `--rescale_learned_sigmas` and `--rescale_timesteps`: Rescaling options (default: **False**).

#### `TRAIN_FLAGS`
Training-specific hyperparameters:

- `--lr`: Learning rate (default: **3e-4**).
- `--batch_size`: Batch size per GPU (default: **32**).
- `--version`: Version name for logging (default: **test**).
- `--log_interval` and `--save_interval`: Interval for logging and saving checkpoints (default: **5**).
- `--dataset`: Dataset to use (default: **prism**).
- `--wavelet`: Use wavelet transformation (default: **True**).

---

Ensure that `--data_dir` points to the correct dataset directory. The default is: /home/linhan/data/PPT_4km_128/


## Sampling
```
bash sample.sh
```

You can adjust parameters in train.sh and sample.sh when you need to.
