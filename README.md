# How to Prove Your Model Belongs to You: A Blind-Watermark based Framework to Protect Intellectual Property of DNN

This is a toy example (on MNIST & CIFAR-10) of how to run our framework presented in [https://arxiv.org/abs/1903.01743](https://arxiv.org/abs/1903.01743)

# Quickstart
* Configure the environment
```
Python=3.7
Pytorch=1.2.0
Others are the latest version
```

# Dataset
* Change `args.dataset='mnist or cifar10'` to define the dataset. And the dataset will be auto-downloaded into `/data`.
* If `args.dataset='mnist'`,  the exclusive logo is a sample picked from the `mnist`dataset randomly.
* If `args.dataset='cifar10'`, the exclusive logo is in the `/data/IEEE/logo`. you can also choose other logos.

# Host Model
* If `args.dataset='mnist'`, you can only implement the framework to embed the watermark into `lenet1, lenet3, lenet5`.
* If `args.dataset='cifar10'`, you can implement the framework to emded the watermark into `vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resnet101, preactresnet18, preactresnet34, googlenet, dpn26, mobilenetv2`.

# Results
## Test Loss curve (vgg19)
![]($resource/results_test_4,5_%E7%9C%8B%E5%9B%BE%E7%8E%8B_%E7%9C%8B%E5%9B%BE%E7%8E%8B.png)

## Images (top:origin;  middle:watermark;  below:logo)

![MNIST]($resource/Epoch_98_img.png)

![CIFAR10]($resource/Epoch_99_img.png)

# Notes
* Random initialization and stochastic gradient descent can cause the objective function to find a new local minimum, which means that the result is different each time. Try to change `args.seed` to start another initialization for optimal results.
* Hyperparameter optimization is needed for optimal results (and other tweaks like using high capacity networks). 

# Citation
If you find  blind-watermark based IPP framework useful in your research, please consider to cite the papers:
```BibTeX
@inproceedings{10.1145/3359789.3359801,
author = {Li, Zheng and Hu, Chengyu and Zhang, Yang and Guo, Shanqing},
title = {How to Prove Your Model Belongs to You: A Blind-Watermark Based Framework to Protect Intellectual Property of DNN},
year = {2019},
isbn = {9781450376280},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3359789.3359801},
doi = {10.1145/3359789.3359801},
booktitle = {Proceedings of the 35th Annual Computer Security Applications Conference},
pages = {126–137},
numpages = {12},
keywords = {intellectual property protection, security and privacy, neural networks, blind watermark},
location = {San Juan, Puerto Rico},
series = {ACSAC ’19}
}
```


