# [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/abs/1907.02189)

**Authors:** Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, Zhihua Zhang

## Summary

The researchers above provide a convergence proof for the Federated Averaging (FedAvg) algorithm when the data is not drawn from the same distribution and devices are not guaranteed to stay in for the entire computation length. The only consider objectives that are both _strongly convex_ and _smooth._

### Problem setting

We wish to solve the following problem:

\[
\min_w F(w):=\sum_{k=1}^Np_kF_k(w)
\]

where:

* $N$ is the number of devices
* $p_k$ is the weight (think: importance) of device $k$ such that $p_k\ge0$ and $\sum_{k=1}^Np_k=1$
* $F_k(w)$ is the local objective of device $k$ such that $F_k(w):=\frac{1}{n_k}\sum_{j=1}^{n_k}l(w;x_{k,j})$
    * $l(\cdot;\cdot)$ is a user defined loss function
    * $x_{k,\cdot}$ is a data point on device $k$
    * $n_k$ is the number of data points on device $k$

### Algorithm updates

The FedAvg algorithm aggregates the model (i.e. $w$) from locally updated models which follow this update scheme:

\[
w^k_{t+i+1}\leftarrow w^k_{t+i}-\eta_{t+i}\nabla F_k(w^k_{t+i},\xi^k_{t+i})
\]

for $i=0,1,\dots,E-1$ where:

* $E\ge1$ is the set number of local updates to perform between each aggregation
* $\eta_{t+i}$ is the step-size of SGD
* $\xi^k_{t+i}$ is a sample uniformly chosen from the data available on device $k$

### Contributions

Assuming all of the local objectives are _strongly convex_ and _smooth_, convergence analyses are given for FedAvg when the following assumptions hold:

* The data is not IID
* All of the device are not active
