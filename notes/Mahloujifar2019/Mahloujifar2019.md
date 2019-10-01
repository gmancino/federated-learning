# [Universal Multi-Party Poisoning Attacks](http://proceedings.mlr.press/v97/mahloujifar19a/mahloujifar19a.pdf)

**Authors:** Saeed Mahloujifar, Mohammad Mahmoody, Ameer Mohammed

## Summary

The researchers above focus on proving that it is possible to corrupt any multi-agent system via poisoning attacks.

## Problem

Similar to other FL cases, we want to globally solve

\[
\min_w F(w):=\sum_{k=1}^Np_kF_k(w)
\]

where each $k$ represents an agent (node, device, etc.) in a network.

The goal of this paper is to design a _corruption technique_ that causes a failure in accurately (or correctly) solving the above problem.

## Assumptions

1. The data is Non-IID
1. The adversary has the ability to sample from distributions of all parties
    * **Claim:** realistic in the case of MASSIVE data breach, otherwise, does this make sense??
1. The adversary corrupts in _total variation distance_
