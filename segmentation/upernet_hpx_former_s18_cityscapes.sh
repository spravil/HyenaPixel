#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

# Copy relevant files into repo
bash prepare.sh

# Start training
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 bash mmsegmentation/tools/dist_train.sh mmsegmentation/configs/hpx_former_s18/upernet_hpx-former-s18-160k_cityscapes.py 4