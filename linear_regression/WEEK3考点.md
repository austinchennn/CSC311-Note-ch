> 来源: 用户提供的课件内容（线性回归 Linear Regression 与梯度下降 Gradient Descent 考点梳理，非 csc311notes 网站原文）

# Week 3 考点梳理 — 线性回归与梯度下降

CSC311 这类课程的考试重点不是默写最终公式，而是考察**对矩阵维度的掌控力**以及**从零手推链式法则**的能力。以下是课件中最容易失分的四个考点。

## 考点一：偏置项 $b$ 是如何被吸收进权重向量的 (trick of dummy feature)

原始线性模型是

$$
y^{(i)}=w_1x_1^{(i)}+\dots+w_Dx_D^{(i)}+b
$$

**消除方法**：在输入向量中强制加入一个虚拟特征 (dummy feature) $x_0^{(i)}=1$ ，然后把偏置项 $b$ 作为权重 $w_0$ 吸收进权重向量中。

**结果**：权重向量变成 $\mathbf{w}=[w_0,w_1,\dots,w_D]^\top$ ，输入向量变成 $\mathbf{x}^{(i)}=[1,x_1^{(i)},\dots,x_D^{(i)}]^\top$ ，最终公式化简为纯点积

$$
y^{(i)}=\mathbf{w}^\top\mathbf{x}^{(i)}
$$

**应试提示**：考试时如果遇到带有单独偏置项的题，第一步永远是先构建扩展矩阵 $\mathbf{X}$（即补上全 1 列）。

## 考点二：损失函数的链式法则手推（大题必考）

直接求解法和梯度下降法的核心都是算梯度，这是考试中最喜欢让学生手写的步骤。

对第 $i$ 个数据点，损失函数为 $L=\frac{1}{2}(y^{(i)}-t^{(i)})^2$ ，其中 $y^{(i)}=\mathbf{w}^\top\mathbf{x}^{(i)}$ 。求 $L$ 对任意权重参数 $w_j$ 的偏导数 $\frac{\partial L}{\partial w_j}$ ，按链式法则分三步：

1. **对外层求导**：

$$
\frac{\partial L}{\partial y^{(i)}}=(y^{(i)}-t^{(i)})
$$

2. **对内层求导**：

$$
\frac{\partial y^{(i)}}{\partial w_j}=\frac{\partial}{\partial w_j}(w_0x_0^{(i)}+\dots+w_jx_j^{(i)}+\dots+w_Dx_D^{(i)})=x_j^{(i)}
$$

3. **相乘合并**：

$$
\frac{\partial L}{\partial w_j}=(\mathbf{w}^\top\mathbf{x}^{(i)}-t^{(i)})x_j^{(i)}
$$

将所有维度的偏导数拼回向量形式，得到单个样本的梯度：

$$
\nabla_{\mathbf{w}}L(\mathbf{w})=(\mathbf{w}^\top\mathbf{x}^{(i)}-t^{(i)})\mathbf{x}^{(i)}
$$

## 考点三：求和符号向矩阵乘法的转化（全班最容易死在维度的一步）

推导代价函数梯度时会得到求和式 $\sum_i(\mathbf{w}^\top\mathbf{x}^{(i)}-t^{(i)})\mathbf{x}^{(i)}$ ，如何把它变成简洁的矩阵运算 $\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{t})$ ？

**定义残差**：令标量残差 $r^{(i)}=\mathbf{w}^\top\mathbf{x}^{(i)}-t^{(i)}$ ，求和式变为 $\sum_i r^{(i)}\mathbf{x}^{(i)}$ 。

**维度盘点**：
- $r^{(i)}$ 是一个 $1\times1$ 的标量。
- $\mathbf{x}^{(i)}$ 是一个 $(D+1)\times1$ 的列向量。
- $\mathbf{r}$ 是包含所有残差的 $N\times1$ 列向量。
- $\mathbf{X}$ 是 $N\times(D+1)$ 的数据矩阵。

**矩阵拼图**：目标是得到一个与权重 $\mathbf{w}$ 维度相同的 $(D+1)\times1$ 梯度向量。要用矩阵乘法得到这个维度，必须拿 $\mathbf{X}^\top$ （维度 $(D+1)\times N$ ）去乘 $\mathbf{r}$ （维度 $N\times1$ ）。

**结论**：$\sum_i r^{(i)}\mathbf{x}^{(i)}$ 等价于 $\mathbf{X}^\top\mathbf{r}$ ，展开即 $\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{t})$ 。

**应试提示**：考试中只要把维度写在矩阵下方逐一核对，就不会搞错谁左乘谁右乘。

## 考点四：两种算法的计算复杂度对决（选择/辨析题）

既然有直接求解法 (normal equation)，为什么还要用梯度下降？这是两节课的底层逻辑碰撞。

**直接求解的死穴**：直接解公式 $\mathbf{w}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{t}$ 中，最要命的是求逆运算 $(\mathbf{X}^\top\mathbf{X})^{-1}$ 。计算一个 $D\times D$ 矩阵的逆，算法复杂度是 $O(D^3)$ 。当特征数量 $D$ 很大（如高维数据）时，直接计算在算力上极其昂贵甚至不可行。

**梯度下降的破局**：梯度下降每次参数更新

$$
\mathbf{w}\leftarrow\mathbf{w}-\frac{\alpha}{N}\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{t})
$$

只需要做矩阵乘法，单步更新复杂度仅为 $O(ND)$ 。对高维度 $D$ 来说，这种迭代方法要高效廉价得多。

## 总结

这四个考点串成一条完整的推导链：**特征扩展 → 链式求导 → 维度重组 → 复杂度分析**，对应 [[LN3]] 中给出的最终结论 $\mathbf{w}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{t}$ 。
