> 来源: https://www.teach.cs.toronto.edu/~csc311notes/linear_regression/mapping.html

# Feature Mapping — 笔记

## 一、核心定义

**特征映射 / 基展开 (feature mapping / basis expansion)**

一个函数 $\psi(\mathbf{x}):\mathbb{R}^D\rightarrow\mathbb{R}^d$ ，把原始输入向量变换成新的特征表示。用 $\psi(\mathbf{x}^{(i)})$ 替换原始 $\mathbf{x}^{(i)}$ 后再跑线性回归，得到的模型在**变换后的特征空间中是线性的**，但在**原始输入空间中可以表达非线性关系**。

**多项式特征映射 (polynomial feature mapping)**

次数为 $M$ 的映射，把标量输入 $x$ 展开成它的各次幂：

$$
\psi(x)=\begin{bmatrix}1 & x & x^2 & \cdots & x^M\end{bmatrix}^\top\in\mathbb{R}^{M+1}
$$

$M$ （多项式的**次数 / degree**）是一个超参数。对向量输入 $\mathbf{x}\in\mathbb{R}^D$ ，次数- $M$ 映射包含所有总次数不超过 $M$ 的单项式（各特征乘积项），特征数量按 $\binom{D+M}{M}$ 增长， $D$ 或 $M$ 增大时会迅速膨胀。

**多项式回归 (polynomial regression)**

以多项式特征为输入的线性回归，拟合

$$
y=w_0+w_1x+w_2x^2+\dots+w_Mx^M=\sum_{j=0}^M w_jx^j=\psi(x)^\top\mathbf{w}
$$

**正弦（傅里叶）映射 (sinusoidal / Fourier mapping)**

用不同频率的 $\sin,\cos$ 作为基函数，适合有周期性结构的数据（如按年变化的温度）。

## 二、公式（举例）

二维输入、次数 $M=2$ 的多项式映射：

$$
\psi([x_1,x_2])=\begin{bmatrix}1\\x_1\\x_2\\x_1x_2\\x_1^2\\x_2^2\end{bmatrix}\in\mathbb{R}^6
$$

正弦映射：

$$
\psi(x)=\begin{bmatrix}1 & \sin(2\pi x) & \cos(2\pi x) & \sin(4\pi x) & \cos(4\pi x) & \cdots\end{bmatrix}^\top
$$

## 三、核心概念与方法论

- **关键洞察**：多项式回归看起来是"非线性模型"，但本质仍是线性回归——只是换了一套（非线性构造出来的）特征，模型 $y=\psi(x)^\top\mathbf{w}$ 依旧**线性于参数** $\mathbf{w}$ ，因此可以照搬同一套最小二乘 / 梯度下降流程，只需把 $\mathbf{x}^{(i)}$ 替换成 $\psi(\mathbf{x}^{(i)})$ 。
- **模块化思想的体现**：特征映射改变的是"数据/预处理步骤"，而不是学习算法本身；换一个 $\psi$ 就能让同一套线性回归适配多项式曲线、周期信号等各种非线性模式，优化器和损失函数完全不用变。
- **模型复杂度与泛化（呼应欠拟合/过拟合）**：次数 $M$ 直接控制模型复杂度—— $M$ 太小（如 $M=0,1$ ）→ 欠拟合 (underfitting)，模型太简单、偏差大，训练和新数据上误差都高； $M$ 太大（如数据点很少时取 $M=9$ ）→ 过拟合 (overfitting)，模型能穿过几乎每个训练点，训练误差趋近于零，但学到的是这批数据特有的噪声，对新数据泛化差。
- **方法论**： $M$ 的选取是典型的**超参数调优**问题——应使用**验证集**而非训练集来挑选 $M$ ，以避免奖励"死记硬背训练数据"的模型。

## 四、例子（精简保留）

- **正弦波数据**：从 $\sin(x)$ 加高斯噪声采样得到的数据点呈明显曲线，直线无法很好拟合，是引出特征映射的动机示例。
- **一维多项式映射表**：输入 $x$ 展开为 $[x^0,x^1,x^2,x^3]$ 四个新特征，目标值 $t$ 不变，用这 4 个新特征做普通线性回归。
