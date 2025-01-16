# Neural Honeytrace: A Robust Plug-and-Play Watermarking Framework against Model Extraction Attack

This is an official implementation of the paper *Neural Honeytrace: A Robust Plug-and-Play Watermarking Framework against Model Extraction Attack.*

**Abstract**: Developing high-performance deep learning models is resource-intensive, leading model owners to utilize Machine Learning as a Service (MLaaS) platforms instead of publicly releasing their models. However, malicious users may exploit query interfaces to execute model extraction attacks, reconstructing the target model's functionality locally. While prior research has investigated triggerable watermarking techniques for asserting ownership, existing methods face significant challenges: (1) most approaches require additional training, resulting in high overhead and limited flexibility, and (2) they often fail to account for advanced attackers, leaving them vulnerable to adaptive attacks.

In this paper, we propose Neural Honeytrace, a robust plug-and-play watermarking framework against model extraction attacks. We first formulate a watermark transmission model from an information-theoretic perspective, providing an interpretable account of the principles and limitations of existing triggerable watermarking. Guided by the model, we further introduce: (1) a similarity-based training-free watermarking method for plug-and-play and flexible watermarking, and (2) a distribution-based multi-step watermark information transmission strategy for robust watermarking. Comprehensive experiments on four datasets demonstrate that Neural Honeytrace outperforms previous methods in efficiency and resisting adaptive attacks. Neural Honeytrace reduces the average number of samples required for a worst-case t-Test-based copyright claim from $12,000$ to $200$ with zero training cost.

![Overview of Neural Honeytrace.](https://github.com/NeurHT/NeurHT/blob/main/Fig_3.png)

## Acknowledgement

The implementation is based on [ModelGuard](https://github.com/Yoruko-Tang/ModelGuard.git), we thank the authors for the high quality of their code.

## Environment and Dataset

Please refer to [ModelGuard](https://github.com/Yoruko-Tang/ModelGuard.git) for environment and dataset preperation.

We will use the following datasets in total in our experiments. Three datasets (CIFAR100, CIFAR10, SVHN) can be automatically downloaded when executing scripts. However, you still need to download **all** the following datasets into ```./data/``` (create it if it does not exist) and unzip them before running any codes. (You can change the default dataset path by changing the ```root``` parameter in the dataset files such as ```./defenses/datasets/caltech256.py```.)

1. [Caltech256](https://data.caltech.edu/records/nyy15-4j048)
2. [CUB200](https://data.caltech.edu/records/65de6-vp158)
3. [ImageNet1k (ILSVRC2012)](http://image-net.org/download-images)
4. [TinyImageNet200](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
5. [Indoor67](http://web.mit.edu/torralba/www/indoor.html)

For Caltech256, CUB200, TinyImageNedt200, Indoor67 and ImageNet1k, you can also use the scripts ```dataset.sh``` to download and unzip in shell:
```shell
sh dataset.sh
```
You can run the following commands to create a new environments for running the codes with Anaconda:
```shell
conda env create -f environment.yml
conda activate neurht
```
**Notice:** Different GPUs may require different versions of PyTorch. Please follow the instructions on the [official website of PyTorch](https://pytorch.org/get-started/locally/) if there is any problem with installing PyTorch. For us, Python=3.9.5, PyTorch=2.0.1+cu117, and CUDA=12.0 worked well with NVIDIA RTX 4090 GPUs.


## Instructions to Run the Codes

We provide automatic scripts to generate the results in our paper. For instance, after preparing all datasets, using the following command will run all the experiments in Table 2. 

```shell
python scripts/run_cifar10.py
```
