# Week 3 — Linear Regression

错题记录：

## 1. 一个 epoch 里有多少次 mini-batch 更新（N/B mini-batch updates）

**题目**：用小批量随机梯度下降 (mini-batch SGD) 训练时，训练集有 $N$ 个样本，批大小 (batch size) 为 $B$ ，一个 epoch 里做多少次参数更新？"N/B mini-batch updates" 是什么意思？

**答案**：一个 epoch 做 $N/B$ 次 mini-batch 更新（ $N$ 不能被 $B$ 整除时为 $\lceil N/B \rceil$ ）。

**解析**：

#### 训练周期 (epoch)

把整个训练集完整过一遍叫一个 epoch。设训练集大小为 $N$ （样本数），批大小为 $B$ （每次更新用的样本数， $1\le B\le N$ ）。

#### 小批量更新 (mini-batch update)

每次从训练集中取出 $B$ 个样本组成一个 mini-batch $\mathcal{M}$ ，只用它们估计梯度并更新一次参数：

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\alpha}{B}\sum_{i\in\mathcal{M}} \nabla_{\mathbf{w}} \mathcal{L}\big(y^{(i)}, t^{(i)}\big)$$

其中 $\alpha$ 是学习率 (learning rate)。

#### 为什么是 N/B

打乱数据后把 $N$ 个样本切成互不重叠的若干块，每块 $B$ 个，共 $N/B$ 块。每块做一次更新，所以一个 epoch 共 $N/B$ 次更新。

- $B=N$ ：全批量梯度下降 (batch gradient descent)，每个 epoch 只更新 **1** 次；
- $B=1$ ：随机梯度下降 (stochastic gradient descent, SGD)，每个 epoch 更新 **$N$** 次；
- $1<B<N$ ：mini-batch SGD，每个 epoch 更新 **$N/B$** 次。

**要点**：每次更新的计算量约为 $O(B)$ ，所以一个 epoch 的总计算量都约为 $O(N)$ 。 $B$ 越小，每个 epoch 更新次数越多，但每次的梯度估计噪声越大（方差越大）。 $B$ 越大，梯度估计越准，但每个 epoch 的更新次数越少。
