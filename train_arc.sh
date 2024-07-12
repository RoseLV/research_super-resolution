#!/bin/bash

#SBATCH --account=stgnn
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1 
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate dgx 
pip install -e .

MODEL_FLAGS="--num_channels 96 --num_res_blocks 2 --learn_sigma True --class_cond False --large_size 128 --small_size 32"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --batch_size 64 --version iDDPM_gamma_small_cos --log_interval 5 --save_interval 5"

python scripts/super_res_train_lightning.py --data_dir /home/linhan/yinlin/data/PPT_4km_128/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
