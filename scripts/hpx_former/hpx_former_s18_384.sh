#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G

DATA_PATH=/data/imagenet
INIT_CKPT=/path/to/hpx_former_s18.pth

ALL_BATCH_SIZE=1024
NUM_GPU=8
GRAD_ACCUM_STEPS=8 # Adjust ac cording to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

python -m torch.distributed.launch --master_port=25500 --nproc_per_node=$NUM_GPU train.py $DATA_PATH \
    --model hpx_former_s18_384 --drop-path 0.2 --model-kwargs head_dropout=0.4 \
    --img-size 384 --epochs 30 --opt adamw --weight-decay 0.05 --lr 5e-5 --sched None --amp --warmup-epochs 0 \
    -b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
    --mixup 0 --cutmix 0 --mixup-prob 1.0 --mixup-switch-prob 0.5 --smoothing 0.1 \
    --aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.25 \
    --model-ema --model-ema-decay 0.9999 --crop-pct 1.0 \
    --initial-checkpoint $INIT_CKPT
