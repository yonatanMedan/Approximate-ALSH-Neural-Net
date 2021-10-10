# ANNI: Approximating Artificial Neural Network Inference With Maximum Inner Product Search

## Introduction
This Package is an realization of https://github.com/yonatanMedan/Approximate-ALSH-Neural-Net/blob/cleaning/Approximating-NN-With_ALSH.pdf written by Yonatan Medan.
In this package we use ```k-Maximum Inner Product Search (k-MIPS)``` to approximate inference of Neural network `. It uses an implementation of ```H2_ALSH``` from the paper as follows:

```bash
Qiang Huang, Guihong Ma, Jianlin Feng, Qiong Fang, and Anthony K. H. Tung. Accurate and Fast
Locality-Sensitive Hashing Scheme for Maximum Inner Product Search. In Proceedings of the 24th
ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining, pages 1561-1570, 2018.
```
Code for H2_ALSH is taken from https://github.com/HuangQiang/H2_ALSH.
## Compilation

The package requires ```g++``` with ```c++11``` support. To download and compile the code, type:

```bash
git clone https://github.com/yonatanMedan/Approximate-ALSH-Neural-Net.git
cd Approximate-ALSH-Neural-Net
make
```
to get the results for mnist:


```bash
Usage: ./approximate-nn-minst -topk <number of active neurons per sign>

```

For example for results of Approximated Neural Network with `10` active neurons use:
```bash
./approximate-nn-minst -topk 5
```
Important! Notice we use 5 as 5 neurons will be used to find the top 5 largest activations and 5 neurons will be used to find the top 5 smallest activations, Summing maximal number of 10 active neurons.
 

## Related Publication
H2_LSH
```bib
@inproceedings{huang2018accurate,
    title={Accurate and Fast Asymmetric Locality-Sensitive Hashing Scheme for Maximum Inner Product Search}
    author={Huang, Qiang and Ma, Guihong and Feng, Jianlin and Fang, Qiong and Tung, Anthony KH},
    booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
    pages={1561--1570},
    year={2018},
    organization={ACM}
}
```
