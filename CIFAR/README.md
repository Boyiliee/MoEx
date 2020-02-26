# MoEx on CIFAR

We train PyramidNet-200 with simple MoEx based on [Cutmix Official Code](https://github.com/clovaai/CutMix-PyTorch). 

## Requirements
This code was tested on the following versions, but they may not be necessary.
```
Python3
torch>=1.0
torchvision>=0.2
Numpy
```

## Training a CIFAR with MoEx 
Here we show an example of training Pyramidnet-200 with MoEx using PONO to extract the moments. The exchange probability $p$ is set to 0.5 and the interpolation weight $\lambda$ is set to 0.2. Please refer to [run.sh](https://github.com/Boyiliee/MoEx/blob/master/CIFAR/run.sh) for details.

```sh
bash run.sh
```


