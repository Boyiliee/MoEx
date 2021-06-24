# MoEx (Moment Exchange)
The official PyTorch implementation of the paper [On Feature Normalization and Data Augmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_On_Feature_Normalization_and_Data_Augmentation_CVPR_2021_paper.pdf).

CVPR 2021

#### Authors: 
* [Boyi Li](https://sites.google.com/site/boyilics/home)*
* [Felix Wu](https://scholar.google.com.tw/citations?user=sNL8SSoAAAAJ&hl=en)*
* [Ser-Nam Lim](https://www.linkedin.com/in/sernam/)
* [Serge Belongie](https://vision.cornell.edu/se3/people/serge-belongie/)
* [Kilian Q. Weinberger](http://kilian.cs.cornell.edu/index.html)

*: Equal Contribution

### Overview
This repo contains the PyTorch implementation of Moment Exchange (MoEx), described in the paper [On Feature Normalization and Data Augmentation](https://arxiv.org/abs/2002.11102). For ImageNet and CIFAR experiments, we select [Positional Normalization (PONO)](https://github.com/Boyiliee/PONO) as the feature normalization method. 

![](./figs/fig1.jpg)

#### Usage
Please follow the instructions in the `README.md` in each subfolder to run experiments with MoEx on [CIFAR](./CIFAR), [ImageNet](./ImageNet), and [ModelNet10/40](./ModelNet).

### Explorations beyond our paper
#### Methods for COVID‐19
* A Cascade‐SEME network for COVID‐19 detection in chest c‐ray images: [Paper](https://aapm.onlinelibrary.wiley.com/doi/full/10.1002/mp.14711?casa_token=y2QQkxTWYD0AAAAA%3AXwClYvwCDmRd4djy_i5Ps0qL64R5mwqlTRhGJCqn-pvzaxsTGuS9C_5f4wlb6M1-jAEJKftiU9BwwZfe)

More information and relevant applications will be updated.

If you find this repo useful, please cite:
```
@inproceedings{li2021feature,
  title={On feature normalization and data augmentation},
  author={Li, Boyi and Wu, Felix and Lim, Ser-Nam and Belongie, Serge and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12383--12392},
  year={2021}
}

@inproceedings{li2019positional,
  title={Positional Normalization},
  author={Li, Boyi and Wu, Felix and Weinberger, Kilian Q and Belongie, Serge},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1620--1632},
  year={2019}
}
```
