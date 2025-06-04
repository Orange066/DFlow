# Context-Aware Iteration Policy Network for Efficient Optical Flow Estimation

This is the official implementation of the paper [Context-Aware Iteration Policy Network for Efficient Optical Flow Estimation](https://arxiv.org/pdf/2312.07180), AAAI 2024.

## Setup

* Clone this repository and navigate to Eigen-Metabolism folder

```
git clone https://github.com/Orange066/DFlow.git
cd DFlow
```

* We use Anaconda to create enviroment.

```
conda create -n ugsp python=3.9
conda activate ugsp
```

* Install Pytorch. 

```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

* Install Python Packages: 

```
pip install -r requirements.txt
```

## Required Data

To evaluate/train RAFT, you will need to download the required datasets. 

* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)

* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

* [Sintel](http://sintel.is.tue.mpg.de/)

* [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)

* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)

  In the files `DFlowFormer/core/datasets.py`, `DGMA/core/datasets.py`, `DKPAFlow/core/datasets.py`, and `DRAFT/core/datasets.py`, you should create symbolic links to wherever the datasets were downloaded.

## Download the Pretrained Model

Please download our pretrained model from the [Hugging Face](https://huggingface.co/Orange066/DFlow_Models) and extract it into the root folder. The data path should be structured as follows:

```
DFlow/
	DFlowFormer/
		checkpoints/
	DGMA/
		results/
	DKPAFlow/
		checkpoints/
	DRAFT/
		checkpoints/
```

## Evaluation

Run the following codes:

```
# test DRAFT
cd DRAFT
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model=checkpoints/raft-things.pth --dataset=kitti
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model=checkpoints/raft-things.pth --dataset=sintel

# test DGMA
cd DGMA
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model ./results/things/gma/gma-things.pth  --dataset kitti
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model ./results/things/gma/gma-things.pth  --dataset sintel

# test DFlowFormer
cd DFlowFormer
CUDA_VISIBLE_DEVICES=0 python evaluate_FlowFormer_tile.py --eval kitti_validation --model checkpoints/things_kitti.pth
CUDA_VISIBLE_DEVICES=0 python evaluate_FlowFormer_tile.py --eval sintel_validation --model checkpoints/things.pth

# test DKPA-Flow
cd DKPAFlow
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model=./checkpoints/kpa-things-kitti.pth  --dataset=kitti
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model=./checkpoints/kpa-things.pth  --dataset=sintel
```

## Credit

Our code borrows from [RAFT](https://github.com/princeton-vl/RAFT/tree/master), [GMA](https://github.com/zacjiang/GMA), [Flowformer](https://github.com/drinkingcoder/FlowFormer-Official) and [KPA-Flow](https://github.com/megvii-research/KPAFlow).

