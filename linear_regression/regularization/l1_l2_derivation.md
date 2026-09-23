> 来源: 补充内容（非课程原文）；相关课程笔记见 [regularization.md](regularization.md)

# L1 / L2 Regularization Derivation — 笔记

面试中"推导 L1、L2 正则化"通常希望从**贝叶斯（最大后验估计 MAP）**角度证明，其次是从**约束优化（拉格朗日乘子法）**角度解释。

## 一、贝叶斯角度：正则化 = 给参数加先验 (prior)

- 不加正则化 → 最大似然估计 (maximum likelihood estimation, MLE)；加正则化 → 最大后验估计 (maximum a posteriori, MAP)。
- 记训练输入为 $X$ 、目标为 $Y$ ，参数为 $W$ （ $n$ 个分量 $w_1,\dots,w_n$ ）。由贝叶斯定理，后验概率为

$$
P(W\mid X,Y)=\frac{P(Y\mid X,W)\,P(W)}{P(Y\mid X)}
$$

- 分母 $P(Y\mid X)$ 与 $W$ 无关，最大化后验只需最大化分子；再取负对数，把"最大化概率"变成"最小化损失"：

$$
\arg\max_W P(W\mid X,Y)=\arg\max_W\bigl[P(Y\mid X,W)\,P(W)\bigr]=\arg\min_W\bigl[\underbrace{-\log P(Y\mid X,W)}_{\text{原损失（MSE / 交叉熵）}}\;\underbrace{-\log P(W)}_{\text{正则化项}}\bigr]
$$

  所以推导的关键就是：**选不同的先验 $P(W)$ ，看 $-\log P(W)$ 长什么样。**

### L2 正则化 ⇔ 高斯先验 (Gaussian prior)

设各 $w_i$ 独立同分布于均值 $0$ 、方差 $\sigma^2$ 的高斯分布：

$$
P(w_i)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{w_i^2}{2\sigma^2}\right),\qquad P(W)=\prod_{i=1}^{n}P(w_i)
$$

取负对数（乘积变求和）：

$$
-\log P(W)=-\sum_{i=1}^{n}\left(\log\frac{1}{\sqrt{2\pi}\sigma}-\frac{w_i^2}{2\sigma^2}\right)=\frac{1}{2\sigma^2}\sum_{i=1}^{n}w_i^2+\text{const}
$$

令 $\lambda=\frac{1}{2\sigma^2}$ 并丢掉与 $W$ 无关的常数，得到 $\lambda\sum_i w_i^2=\lambda\lVert W\rVert_2^2$ 。

**结论**：L2 正则化等价于"参数服从零均值高斯先验"的 MAP。先验方差 $\sigma^2$ 越小（越相信权重接近 0）， $\lambda$ 越大、正则越强。（与 `linear_regression/regularization.md` 中的 $\frac{1}{2}\lVert\mathbf{w}\rVert_2^2$ 只差一个常数因子，可吸收进 $\lambda$ 。）

### L1 正则化 ⇔ 拉普拉斯先验 (Laplace prior)

设各 $w_i$ 独立同分布于均值 $0$ 、尺度参数 (scale) $b$ 的拉普拉斯分布（此处 $b$ 不是偏置）：

$$
P(w_i)=\frac{1}{2b}\exp\left(-\frac{\lvert w_i\rvert}{b}\right)
$$

$$
-\log P(W)=-\sum_{i=1}^{n}\left(\log\frac{1}{2b}-\frac{\lvert w_i\rvert}{b}\right)=\frac{1}{b}\sum_{i=1}^{n}\lvert w_i\rvert+\text{const}
$$

令 $\lambda=\frac{1}{b}$ ，得到 $\lambda\sum_i\lvert w_i\rvert=\lambda\lVert W\rVert_1$ 。

**结论**：L1 正则化等价于"参数服从零均值拉普拉斯先验"的 MAP。

## 二、约束优化角度：正则化 = 限制参数空间

正则化本质上是把参数限制在一个有界区域内（结构风险最小化 (structural risk minimization)），设原损失为 $L(W)$ ：

$$
\min_W L(W)\quad\text{s.t.}\quad\lVert W\rVert_1\le c\ \ \text{（L1）}\quad\text{或}\quad\lVert W\rVert_2^2\le c\ \ \text{（L2）}
$$

用拉格朗日乘子法 (Lagrange multipliers) 转为无约束问题，以 L1 为例：

$$
J(W,\lambda)=L(W)+\lambda\bigl(\lVert W\rVert_1-c\bigr)
$$

对 $W$ 最小化时 $-\lambda c$ 是常数可忽略，剩下 $L(W)+\lambda\lVert W\rVert_1$ （L1 正则）；把约束换成 $\lVert W\rVert_2^2\le c$ 同理得到 $L(W)+\lambda\lVert W\rVert_2^2$ （L2 正则）。约束半径 $c$ 越小，对应的 $\lambda$ 越大。

## 三、高频追问：为什么 L1 产生稀疏解 (sparse solution)，L2 通常不会？

1. **几何角度**：把损失等高线和约束区域画在一起。
   - L1 的约束区域是菱形/方形，顶点在坐标轴上；等高线向外扩张时大概率先碰到顶点，此时某些 $w_i=0$ → 稀疏解。
   - L2 的约束区域是圆形，切点一般落在圆弧上，各分量通常都不为 $0$ 。
2. **先验密度角度**（接上面的推导）：
   - 拉普拉斯分布在 $w=0$ 处有尖峰，参数恰好取 $0$ 的倾向远强于其它值 → 倾向于把权重压到 $0$ 。
   - 高斯分布在 $w=0$ 附近是平滑的钟形顶部，只让权重变小（权重衰减 (weight decay)），但不会恰好等于 $0$ 。
