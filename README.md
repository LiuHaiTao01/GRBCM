Aggregation GPs for large-scale regression
====

This is the implementation of the scalable GP developed in "*[Haitao Liu, Jianfei Cai, Yi Wang, Yew-Soon Ong, Generalized Robust Bayesian Committee Machine for Large-scale Gaussian Process Regression, 2018 ICML](https://arxiv.org/abs/1806.00720)*". Please see the paper for further details.

We implement state-of-the-art aggregation models including 
* product-of-experts (PoE),
* generalized PoE (GPoE),
* Bayesian committee machine (BCM),
* robust BCM (RBCM),
* generalized RBCM (GRBCM), 
* nested pointwise aggregation of experts (NPAE).

This code relies on the "[GPML toolbox](http://www.gaussianprocess.org/gpml/code/matlab/doc/)".

To run the example file, execute:
```
demo_toy.m
```


# GRBCM
