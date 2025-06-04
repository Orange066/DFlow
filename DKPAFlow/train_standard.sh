#!/bin/bash
mkdir -p checkpoints
#CUDA_VISIBLE_DEVICES=4,5 python -u train.py --name raft-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 100000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001
#CUDA_VISIBLE_DEVICES=4,5 python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001
#CUDA_VISIBLE_DEVICES=4,5 python -u train.py --name raft-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85
#CUDA_VISIBLE_DEVICES=4,5 python -u train.py --name raft-kitti  --stage kitti --validation kitti --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 1 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85

#CUDA_VISIBLE_DEVICES=2,3 python -u train.py --name raft-things --stage things --validation kitti --restore_ckpt raft-things.pth --gpus 0 1 --num_steps 40000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001



CUDA_VISIBLE_DEVICES=0,3 python -u train.py --name kpa-things --stage things --validation kitti --restore_ckpt KPAFlow-CT2S.pth --gpus 0 1 --num_steps 40000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --dataset sintel
CUDA_VISIBLE_DEVICES=0,3 python -u train.py --name kpa-sintel --stage sintel --validation sintel --restore_ckpt KPAFlow-CTSKH2S.pth --gpus 0 1 --num_steps 40000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85 --dataset sintel
CUDA_VISIBLE_DEVICES=1,2 python -u train.py --name kpa-kitti  --stage kitti --validation kitti --restore_ckpt KPAFlow-CTSKH2K.pth --gpus 0 1 --num_steps 20000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --dataset kitti


