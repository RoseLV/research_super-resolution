#!/bin/bash

export OPENAI_LOGDIR=log
MODEL_FLAGS="--num_channels 128 --num_res_blocks 2 --learn_sigma True --class_cond False --large_size 64 --small_size 32"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --batch_size 8"

python scripts/super_res_train.py --data_dir datasets/valid_64x64/valid_64x64/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
