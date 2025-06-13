# NIReID
The official repository for Occluded Person Re-Identification with Noise Injection.

## Prepare Datasets
Download the person datasets datasets.

Then unzip them and rename them under your "dataset_root" directory like
```bash
dataset_root
├── Occluded_Duke
├── P-DukeMTMC-reid
├── Market-1501-v15.09.15
├── DukeMTMC-reID
├── MSMT17
└── cuhk03-np
```

### Train on Occluded_Duke
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset occluded_duke --noise-dataset veri776 --paste-dataset veri776 --noise-interval 10 --height-ratio 0.4 --gpus 0 --epochs 5,75 --erasing 0.40 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
### Train on P-DukeMTMC
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset p_dukemtmc --noise-dataset veri776 --paste-dataset veri776 --noise-interval 10 --height-ratio 0.2 --gpus 0 --epochs 5,75 --erasing 0.40 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
### Train on Market1501
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset market1501 --noise-dataset veri776 --paste-dataset veri776 --noise-interval 10 --height-ratio 0.1 --gpus 0 --epochs 5,75 --erasing 0.40 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
### Train on DukeMTMC
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset dukemtmc --noise-dataset veri776 --paste-dataset veri776 --noise-interval 10 --height-ratio 0.1 --gpus 0 --epochs 5,75 --erasing 0.40 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
### Train on CUHK03 Detected
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset npdetected --noise-dataset veri776 --paste-dataset veri776 --noise-interval 10 --height-ratio 0.1 --gpus 0 --epochs 5,155 --erasing 0.40 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
### Train on CUHK03 Labeled
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset nplabeled --noise-dataset veri776 --paste-dataset veri776 --noise-interval 10 --height-ratio 0.1 --gpus 0 --epochs 5,155 --erasing 0.40 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```
### Train on MSMT17
```bash
python train.py --net regnet_y_1_6gf --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset msmt17 --noise-dataset veri776 --paste-dataset veri776 --noise-interval 10 --height-ratio 0.1 --gpus 0 --epochs 5,75 --erasing 0.40 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25
```