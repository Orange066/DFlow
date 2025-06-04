# C + T -> K
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model=KPAFlow-CT2K.pth  --dataset=kitti

CUDA_VISIBLE_DEVICES=0 python evaluate.py --model=./checkpoints/kpa-kitti.pth  --dataset=kitti

CUDA_VISIBLE_DEVICES=0 python evaluate.py --model=/home/user3/KPAFlow-main-incre-8/checkpoints_0/kpa-kitti.pth  --dataset=kitti
