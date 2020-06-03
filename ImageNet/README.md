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
bash run_moex_resnet.sh
```
where `$DATA` is the path to the ImageNet data folder, `$SAVE` is path to save the model, `$NPUS` is the number of GPUs to use, and `$WORKERS_PER_GPU` is the number of workers to pre-process the data for each GPU. Please adjust the batch size according to your GPU memory. For example, you may use `-b 128` or `-b 64`. The learning rate is adjusted based on the batch size.

## Training a ResNet-50 with CutMix + MoEx 
```sh
bash run_moex+cutmix_resnet.sh
```

## Training a DenseNet-265 with MoEx
```sh
bash run_moex_densenet.sh
```

## Model Zoo
[Moex+cutmix_resnet50_20.9](https://drive.google.com/file/d/1cCvhQKV93pY-jj8f5jITywkB9EabiQDA/view?usp=sharing). 

[Moex_densenet265_20.9](https://drive.google.com/file/d/1qzGORRZ1GLvTZLHj4UlQIvhD2F11aVZ8/view?usp=sharing). 

