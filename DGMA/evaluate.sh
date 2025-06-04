#!/usr/bin/env bash
#python evaluate.py --model checkpoints/gma-chairs.pth --dataset chairs
CUDA_VISIBLE_DEVICES=6 python evaluate.py --model checkpoints_pretrained/gma-things.pth --dataset kitti
#python evaluate.py --model checkpoints/gma-sintel.pth --dataset sintel
#python evaluate.py --model checkpoints/gma-kitti.pth --dataset kitti