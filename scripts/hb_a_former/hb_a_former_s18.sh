#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G

DATA_PATH=/data/imagenet

ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS

python -m torch.distributed.launch --master_port=25500 --nproc_per_node=$NUM_GPU train.py $DATA_PATH \
    --model hb_a_former_s18 --drop-path 0.2 --model-kwargs head_dropout=0.0 \
    --opt adamw --lr 4e-3 --weight-decay 0.05  --sched cosine --warmup-lr 1e-6 --min-lr 1e-5 --amp \
    --epochs 300 --warmup-epochs 20 --cooldown-epochs 10 \
    -b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
    --mixup 0.8 --cutmix 1.0 --mixup-prob 1.0 --mixup-switch-prob 0.5 --smoothing 0.1 \
    --aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.25