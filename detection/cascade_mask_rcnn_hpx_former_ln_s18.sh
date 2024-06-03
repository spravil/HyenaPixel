#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=768G

# Copy relevant files into repo
bash prepare.sh

# Start training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 bash mmdetection/tools/dist_train.sh mmdetection/configs/hpx_former_s18/cascade-mask-rcnn_hpx-former-ln-s18-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py 8
