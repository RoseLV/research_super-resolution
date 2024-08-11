#!/bin/bash

# MODEL_FLAGS="--num_channels 128 --num_res_blocks 2 --learn_sigma True --class_cond False --large_size 128 --small_size 16"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
# TRAIN_FLAGS="--lr 3e-4 --model_dir checkpoints --log_interval 5 --save_interval 5"
# 
# SAMPLE_FLAGS="--batch_size 64 --num_samples -1 --timestep_respacing 250 --dataset prism"
# 
# python scripts/super_res_sample_new.py --model_path ./logs/checkpoints/prism_128_16/ema_0.9999_113400.pt --data_dir /home/linhan/data/PPT_4km_128/ $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS 



# MODEL_FLAGS="--num_channels 128 --num_res_blocks 2 --learn_sigma True --class_cond False --large_size 128 --small_size 32"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
# TRAIN_FLAGS="--lr 3e-4 --model_dir checkpoints --log_interval 5 --save_interval 5"
# 
# SAMPLE_FLAGS="--batch_size 64 --num_samples -1 --timestep_respacing 250 --dataset prism"
# python scripts/super_res_sample_new.py --model_path ./logs/checkpoints/prism_128_32/ema_0.9999_113400.pt --data_dir /home/linhan/data/PPT_4km_128/ $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS 



# MODEL_FLAGS="--num_channels 128 --num_res_blocks 2 --learn_sigma True --class_cond False --large_size 128 --small_size 16"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
# TRAIN_FLAGS="--lr 3e-4 --model_dir checkpoints --log_interval 5 --save_interval 5"

# SAMPLE_FLAGS="--batch_size 64 --num_samples -1 --timestep_respacing 250 --dataset prism --guidance_scale 0.1"

# python scripts/super_res_sample_new.py --model_path ./logs/checkpoints/prism_128_16_topo/ema_0.9999_113400.pt --data_dir /data/share/PPT_4km_128/ --topo_file /data/share/topo_4km_128.tiff --guidance_scale 5 --output_dir ./logs/ $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS 


#!/bin/bash

MODEL_FLAGS="--num_channels 128 --num_res_blocks 2 --learn_sigma True --class_cond False --large_size 128 --small_size 16"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --model_dir checkpoints --log_interval 5 --save_interval 5"
SAMPLE_FLAGS="--batch_size 64 --num_samples -1 --timestep_respacing 250 --dataset prism "
# Loop over guidance_scale values from 1 to 100
# for guidance_scale in 0.2 0.3 0.5 0.8 1 1.2 1.5 2 5 10
for scale in 1 2 5 10 20 50 100
do
    for alpha in 0.1 0.5 0.9
    do
        echo "Running with scale=${scale} alpha=${alpha}"

        # Run the python script with the current scale and alpha
        python scripts/super_res_sample_new.py \
            --model_path ./logs/checkpoints/prism_128_16_topo/ema_0.9999_113400.pt \
            --data_dir /data/share/PPT_4km_128/ \
            --topo_file /data/share/topo_4km_128.tiff \
            --guidance_scale 1 \
            --scale ${scale} \
            --alpha ${alpha} \
            $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS
    done
done
