#!/bin/bash

MODEL_FLAGS="--num_channels 96 --num_res_blocks 2 --learn_sigma True --class_cond False --large_size 64 --small_size 32"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --model_dir checkpoints --log_interval 5 --save_interval 5"

SAMPLE_FLAGS="--batch_size 64 --num_samples 64 --timestep_respacing 250 --dataset mlde_single"

python scripts/super_res_sample_new.py --model_path ./logs/mlde_gamma_single_32/ema_0.9999_083300.pt --data_dir /home/linhan/data/bham_gcmx-4x_12em_psl-sphum4th-temp4th-vort4th_eqvt_random-season/ $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS 
