# MoEx on ImageNet
This code is based on [apex's ImageNet example](https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py) and [CutMix's official code](https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py). 

## Requirements
This code was tested on the following versions, but they may not be necessary.
```
torch>=1.3.0
torchvision>=0.4.2
tensorboard>=2.0.1
apex>=0.1
```

## Training a ResNet-50 with MoEx 
Here we show an example of training ResNet-50 with MoEx using PONO to extract the moments.
The exchange probability $p$ is set to 1 and the interpolation weight $\lambda$ is set to 0.9.

```sh
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
    $DATA --print-freq 20 --workers $WORKERS_PER_GPU --opt-level O1 \
    -a moex_resnet50 -b 224 --epochs 300 \
    --moex_norm pono --moex_lam 0.9 --moex_prob 1 \
    --output_dir $SAVE
```
where `$DATA` is the path to the ImageNet data folder, `$SAVE` is path to save the model, `$NPUS` is the number of GPUs to use, and `$WORKERS_PER_GPU` is the number of workers to pre-process the data for each GPU. Please adjust the batch size according to your GPU memory. For example, you may use `-b 128` or `-b 64`. The learning rate is adjusted based on the batch size.

## Training a ResNet-50 with CutMix + MoEx 
```sh
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
    $DATA --print-freq 20 --workers $WORKERS_PER_GPU --opt-level O1 \
    -a moex_resnet50 -b 224 --epochs 300 \
    --moex_norm pono --moex_lam 0.95 --moex_prob 0.25 \
    --cutmix_prob 1 \
    --output_dir $SAVE
```

## Training a DenseNet-265 with MoEx
```sh
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py \
    $DATA --print-freq 20 --workers $WORKERS_PER_GPU --opt-level O1 \
    -a moex_densenet265 -b 224 --epochs 300 \
    --moex_layer pool0 --moex_norm pono --moex_lam 0.9 --moex_prob 1 \
    --output_dir $SAVE
```
