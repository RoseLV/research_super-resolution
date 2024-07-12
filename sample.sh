#!/bin/bash

MODEL_FLAGS="--num_channels 128 --num_res_blocks 2 --learn_sigma True --class_cond False --large_size 64 --small_size 32"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --model_dir checkpoints --log_interval 5 --save_interval 5"

SAMPLE_FLAGS="--batch_size 64 --num_samples 64 --timestep_respacing 250"

python scripts/super_res_sample_new.py --model_path ema_0.9999_058200.pt --data_dir /home/linhan/data/PPT_hr_yearly/ $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS 
