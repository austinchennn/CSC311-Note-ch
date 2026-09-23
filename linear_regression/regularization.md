> 来源: https://www.teach.cs.toronto.edu/~csc311notes/linear_regression/regularization.html

# Regularization — 笔记

## 一、核心定义

#### 正则化 (regularization)

对训练过程的一种修改，使模型的选择不完全依据训练误差，还依据其他准则（通常是偏好更简单的模型），从而改善泛化、减少过拟合。

#### 正则化项 / 正则化函数 (regularization term / regularizer) $\mathcal{R}(\mathbf{w})$

直接施加在代价函数上的惩罚项，用来编码"偏好某些模型"的准则；加入正则化项后，优化目标仍是让总代价最低，只是"代价"不再单指 MSE。

#### 正则化后的代价函数

$$
\mathcal{E}_{reg}(\mathbf{w})=\mathcal{E}(\mathbf{w})+\lambda\mathcal{R}(\mathbf{w})
$$

其中 $\mathcal{E}(\mathbf{w})$ 是原始 MSE， $\lambda$ 是控制两项相对重要性的**超参数**。

#### L2 正则化项 (L2 regularization term / L2 regularizer)

$$
\mathcal{R}(\mathbf{w})=\frac{1}{2}\lVert\mathbf{w}\rVert_2^2=\frac{1}{2}\sum_{j=1}^D w_j^2
$$

（偏置 $b$ 通常不参与正则化——回忆 $y=wx+b$ 中 $b$ 只是整体平移，不控制"陡峭程度"，惩罚它没有意义。）L2 是惩罚大权重的常见选择，但不是唯一选择。

## 二、核心概念与思想

- **为何用权重大小衡量复杂度**：观察发现，多项式回归中 $M$ 越大、拟合曲线越"扭曲"，学出来的权重量级也越大；大权重意味着某个特征对预测的影响过大（例如图像分类中某一像素就能大幅改变预测），这通常是过拟合的信号，因此可以用权重的 $L^2$ 范数作为模型复杂度的度量。
- **$\lambda$ 的作用与权衡**： $\lambda$ 太小 → MSE 项主导，正则化不足以抑制过拟合； $\lambda$ 太大 → 正则化项主导，可能导致欠拟合。需要像调其它超参数一样调 $\lambda$ （用验证集）。
- **解耦模型复杂度与数据复杂度**：有了正则化，就不必让 $M$ （模型复杂度）精确匹配数据真实的复杂度——可以直接选一个足够大、注定会过拟合的 $M$ ，再用正则化（调 $\lambda$ ）来控制过拟合程度。这是实践中很常用的策略：宁可模型"过大"，再用正则化去管住它。
- **正则化的普适性**（呼应 Idea #1：学习即优化）：正则化本质是在优化目标里加入我们真正在意的偏好，不限于惩罚过拟合——dropout、数据增强等也是正则化的例子，理论上也可以设计正则化项去促进模型的公平性等其他目标。

## 三、例子（精简保留）

- **十点正弦波拟合实验**：仅用 10 个带噪声的采样点做多项式回归，观察到 $M$ 越大、学到的权重 $\mathbf{w}$ 幅值越大、曲线越剧烈震荡；在 $M=9$ 时可以让训练误差降到零（曲线穿过全部 10 个点），但明显不再像原始的正弦波，是过拟合的直观展示。

## 四、补充：L1 / L2 正则化的推导（面试角度）

> 本节**不是**原文内容，是面试常考的补充推导：通常希望从**贝叶斯（最大后验估计 MAP）**角度证明，其次是从**约束优化（拉格朗日乘子法）**角度解释。

### 1. 贝叶斯角度：正则化 = 给参数加先验 (prior)

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

#### L2 正则化 ⇔ 高斯先验 (Gaussian prior)

设各 $w_i$ 独立同分布于均值 $0$ 、方差 $\sigma^2$ 的高斯分布：

$$
P(w_i)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{w_i^2}{2\sigma^2}\right),\qquad P(W)=\prod_{i=1}^{n}P(w_i)
$$

取负对数（乘积变求和）：

$$
-\log P(W)=-\sum_{i=1}^{n}\left(\log\frac{1}{\sqrt{2\pi}\sigma}-\frac{w_i^2}{2\sigma^2}\right)=\frac{1}{2\sigma^2}\sum_{i=1}^{n}w_i^2+\text{const}
$$

令 $\lambda=\frac{1}{2\sigma^2}$ 并丢掉与 $W$ 无关的常数，得到 $\lambda\sum_i w_i^2=\lambda\lVert W\rVert_2^2$ 。

**结论**：L2 正则化等价于"参数服从零均值高斯先验"的 MAP。先验方差 $\sigma^2$ 越小（越相信权重接近 0）， $\lambda$ 越大、正则越强。（与第一节 $\frac{1}{2}\lVert\mathbf{w}\rVert_2^2$ 只差一个常数因子，可吸收进 $\lambda$ 。）

#### L1 正则化 ⇔ 拉普拉斯先验 (Laplace prior)

设各 $w_i$ 独立同分布于均值 $0$ 、尺度参数 (scale) $b$ 的拉普拉斯分布（此处 $b$ 不是偏置）：

$$
P(w_i)=\frac{1}{2b}\exp\left(-\frac{\lvert w_i\rvert}{b}\right)
$$

$$
-\log P(W)=-\sum_{i=1}^{n}\left(\log\frac{1}{2b}-\frac{\lvert w_i\rvert}{b}\right)=\frac{1}{b}\sum_{i=1}^{n}\lvert w_i\rvert+\text{const}
$$

令 $\lambda=\frac{1}{b}$ ，得到 $\lambda\sum_i\lvert w_i\rvert=\lambda\lVert W\rVert_1$ 。

**结论**：L1 正则化等价于"参数服从零均值拉普拉斯先验"的 MAP。

### 2. 约束优化角度：正则化 = 限制参数空间

正则化本质上是把参数限制在一个有界区域内（结构风险最小化 (structural risk minimization)），设原损失为 $L(W)$ ：

$$
\min_W L(W)\quad\text{s.t.}\quad\lVert W\rVert_1\le c\ \ \text{（L1）}\quad\text{或}\quad\lVert W\rVert_2^2\le c\ \ \text{（L2）}
$$

用拉格朗日乘子法 (Lagrange multipliers) 转为无约束问题，以 L1 为例：

$$
J(W,\lambda)=L(W)+\lambda\bigl(\lVert W\rVert_1-c\bigr)
$$

对 $W$ 最小化时 $-\lambda c$ 是常数可忽略，剩下 $L(W)+\lambda\lVert W\rVert_1$ （L1 正则）；把约束换成 $\lVert W\rVert_2^2\le c$ 同理得到 $L(W)+\lambda\lVert W\rVert_2^2$ （L2 正则）。约束半径 $c$ 越小，对应的 $\lambda$ 越大。

### 3. 高频追问：为什么 L1 产生稀疏解 (sparse solution)，L2 通常不会？

1. **几何角度**：把损失等高线和约束区域画在一起。
   - L1 的约束区域是菱形/方形，顶点在坐标轴上；等高线向外扩张时大概率先碰到顶点，此时某些 $w_i=0$ → 稀疏解。
   - L2 的约束区域是圆形，切点一般落在圆弧上，各分量通常都不为 $0$ 。
2. **先验密度角度**（接上面的推导）：
   - 拉普拉斯分布在 $w=0$ 处有尖峰，参数恰好取 $0$ 的倾向远强于其它值 → 倾向于把权重压到 $0$ 。
   - 高斯分布在 $w=0$ 附近是平滑的钟形顶部，只让权重变小（权重衰减 (weight decay)），但不会恰好等于 $0$ 。
