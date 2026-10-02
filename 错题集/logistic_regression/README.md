# Week 4 — Logistic Regression

错题记录：

## 1. 为什么 CE 能解决 MSE 的梯度消失问题

详见 [为什么CE能解决MSE的vanishing gradient问题.md](为什么CE能解决MSE的vanishing%20gradient问题.md)

## 2. 为什么"线性模型 + 平方误差"做分类不 work

**题目**：为什么用线性模型 + 平方误差 (linear model + squared error) 做二分类不 work？

> Squared error penalizes predictions that are "too correct" (e.g. $z = 10$ for $t = 1$ gives a large loss) since $z$ is unbounded while $t\in\{0,1\}$. Such correctly-classified points still pull the decision boundary and can make it misclassify other points.

**答案**：平方误差会惩罚"太正确"的预测 (too correct)。 $z$ 没有上界，而 $t\in\{0,1\}$ ，所以离边界很远、已经分对的点仍会有很大的损失和梯度。这些点会拖动决策边界，导致别的点被分错。

**解析**：

#### 数学原因：看梯度

设输入 $\mathbf{x}\in\mathbb{R}^{D}$ ，权重 $\mathbf{w}\in\mathbb{R}^{D}$ ， $z=\mathbf{w}^\top\mathbf{x}\in\mathbb{R}$ ，目标 $t\in\{0,1\}$ ，阈值为 0.5（ $z\ge 0.5$ 判为 1）。平方误差 $\mathcal{L}(z,t)=\frac12(z-t)^2$ 对 $\mathbf{w}$ 的梯度为：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}}=\frac{\partial \mathcal{L}}{\partial z}\frac{\partial z}{\partial \mathbf{w}}=(z-t)\,\mathbf{x}$$

这个梯度只看 $z$ 离 $t$ 有多远，不管这个点有没有分对。

- 以 $t=1$ ， $z=10$ 为例：这个点已经分对了（ $10\ge0.5$ ），而且非常自信，但损失 $\mathcal{L}=\frac12(10-1)^2=40.5$ 。
- 更新 $\mathbf{w}\leftarrow\mathbf{w}-9\alpha\mathbf{x}$ （ $\alpha$ 为学习率）会把 $z$ 往 1 拉回去。
- $z$ 无上界而 $t$ 只能取 0/1，所以**分得越对的点， $z-t$ 越大，对 $\mathbf{w}$ 的影响越大**。模型只能减小斜率、移动截距，决策边界 $\mathbf{w}^\top\mathbf{x}=0.5$ 也就跟着移动。

对比交叉熵 (cross-entropy)：梯度为 $(\sigma(z)-t)\mathbf{x}$ 。 $z=10$ 时 $\sigma(z)\approx1$ ，梯度几乎为 0，分对的点基本不再推动边界。

#### 具体例子（一维， $y=w_0+w_1x$ ，阈值 0.5）

最小二乘解： $w_1=\dfrac{\sum_i (t_i-\bar t)x_i}{\sum_i (x_i-\bar x)x_i}$ ， $w_0=\bar t-w_1\bar x$ 。

**数据 A**： $x=(0,1,2,3)$ ， $t=(0,0,1,1)$ 。

- $\bar x=1.5$ ， $\bar t=0.5$ ， $w_1=2/5=0.4$ ， $w_0=0.5-0.6=-0.1$ 。
- 决策边界： $-0.1+0.4x=0.5\Rightarrow x=1.5$ ，四个点全部分对。

**数据 B**：在 A 的基础上加一个很容易分对的点 $x=10$ ， $t=1$ 。

- $\bar x=3.2$ ， $\bar t=0.6$ 。
- $\sum_i(t_i-\bar t)x_i=-0.6+0.8+1.2+4=5.4$ 。
- $\sum_i(x_i-\bar x)x_i=\sum_i x_i^2-N\bar x^2=114-51.2=62.8$ （ $N=5$ ）。
- $w_1=5.4/62.8\approx0.086$ ， $w_0\approx0.6-0.086\times3.2\approx0.325$ 。
- 决策边界： $0.325+0.086x=0.5\Rightarrow x\approx2.04$ 。

| $x$ | 0 | 1 | **2** | 3 | 10 |
|---|---|---|---|---|---|
| $z$ | 0.32 | 0.41 | **0.497** | 0.58 | 1.18 |
| $t$ | 0 | 0 | **1** | 1 | 1 |
| 预测 | 0 | 0 | **0 ✗** | 1 | 1 |

**结论**：新加的点本身分对了。但如果沿用 A 的模型，它的 $z=-0.1+0.4\times10=3.9$ ，平方误差是 $\frac12(3.9-1)^2\approx4.2$ 。为了压低这个损失，最小二乘把斜率从 0.4 压到了 0.086，边界从 1.5 右移到 2.04，原本分对的 $x=2$ 被误分成类别 0。
