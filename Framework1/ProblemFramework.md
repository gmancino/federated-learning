# A Distributed Optimization Framework in the Presence of Adversarial Data

In the context of learning, imagine we want to train some Machine Learning model to learn some optimal parameter $\theta^*$ for the sake of solving a classification problem. The architecture we have for this problem is a set of $n$ agents (nodes, devices, etc.) working collectively to find $\theta^*$ by optimizing over their own training data. This **Federated Learning** setting can be summarized by the graphic below:

![FL Setting](../../notes/Images/FLStructure.png)

### Malicious Data Issue

Imagine that some subset of your workers $\mathcal{K}\subset\mathcal{F}:=\{F_1,F_2,\dots,F_n\}$ have the goal of _corrupting_ your model by injecting training data that does not fit the problem you are actually trying to solve. The question we pose is, **how can you still perform Federated Learning in this setting?**

### Our Assumptions for this Framework

Given a set of $n$ agents, $\mathcal{F},$ each with their own set of data $D_i$ for all $i=1,\dots,n$, where $D_i$ consists of training pairs $\{\textbf{x}_j^i,y_j^i\}$ where $\textbf{x}^i_j\in\mathbb{R}^m$ is the $j^{th}$ training data point on agent $i$ and $y_j^i$ is the corresponding classification label. Note, $j$ will differ depending on which device we are currently considering.

##### Distributional Assumptions

1. Homogeneity in the classification distribution

We assume that each devices' data follows the same conditional distribution regarding classification, i.e.

\[
D_i\sim\mathbb{P}_i(y|x).
\]

This corresponds to saying that if the data is meant to be classified in one category, all data that is "similar" to that data should also be classified in the category -- so the classification distribution _must_ be the same. 

2. Bound on number of malicous agents

We assume there exists some proportion of agents $\rho=|\mathcal{K}|/|\mathcal{F}|$ that are malicious. This way we can use $\rho$ in our calculations explicitly.

3. Distributional "difference" is bounded

For every $i$, _non-malicious_ agent, we assume

\[
d(\mathbb{P}_i(y|x), \mathbb{P}_{true}(y|x))\le\epsilon
\]

where $\mathbb{P}_{true}(y|x)$ is the true, but unknown, classification distribution, $d(\cdot,\cdot)$ is some distance metric, and $\epsilon$ is a small positive number.

This ensures that if you are able to select the non-malicous workers, you can essentially obtain an i.i.d. training set for each local model training.

### Proposed Algorithm

At time, $t=0$ have each agent send statistics (unsure what exactly yet) to the Central Server so that the server can appropriately compute $d(\mathbb{P}_i(y|x), \mathbb{P}_{true}(y|x))$ to find malicious nodes.

* Note: _lots_ of questions here still, i.e. how to estimate $\mathbb{P}_{true}(y|x),$ what is a "good" distance function, how does epsilon depend on desired accuracy, etc.

Once malicious nodes are identified, throw those devices out of the computation. Then run some Federated Learning model with the remaining devices.
