# MoEx on ModelNet
This code is based on the [example](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_classification.py) in PyTorch Geometric. We appreciate the PyTorch Geometric team for sharing their efficient implementation.

## Requirements
This code was tested on the following versions, but they may not be necessary. Please follow the [installation instructions](https://github.com/rusty1s/pytorch_geometric#installation) in PyTorch Geometricl repo to install the packages.
```
torch>=1.3.0
torchvision>=0.4.2
torch-geometric>=1.3.2
```


## Training a PointNet++ on ModelNet10 with MoEx (using instance normalization)
Note: there is a typo in our first draft (arXiv v1) saying that moex_lambda is set to 1 (i.e., not interpolating the labels), but in fact we use 0.9.
```sh
python pointnet2_classification_moex.py --moex_prob 0.5 --moex_lambda 0.9 --moex_norm "in" --data 10
```

## Training a PointNet++ on ModelNet40 with MoEx (using instance normalization)
```sh
python pointnet2_classification_moex.py --moex_prob 0.5 --moex_lambda 0.9 --moex_norm "in" --data 40
```