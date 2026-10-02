> 来源: https://www.teach.cs.toronto.edu/~csc311notes/linear_classification/limitations_of_linear_models.html

# Limitations of Linear Models — 笔记

## 一、核心定义

#### 线性可分 (linearly separable)

两组点存在一个能把它们完全分开的超平面。线性回归、逻辑回归、softmax 回归共享同一个结构性限制——预测都基于输入特征的线性函数，因此分类边界永远只能是超平面；当真实类别边界是曲线或由多个不连通区域组成时，无论用多少训练数据、用什么优化方法，线性模型都无法完美分类。

#### 半空间 (half-space)

$\mathbb{R}^d$ 中形如 $H=\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}+b\geq0\}$ 的点集，即超平面 $\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}+b=0\}$ 一侧的所有点。

#### 凸集 (convex set)

集合 $S$ 中任取两点 $\mathbf{x}^{(a)},\mathbf{x}^{(b)}$ ，连接它们的线段（即所有 $\lambda\mathbf{x}^{(a)}+(1-\lambda)\mathbf{x}^{(b)}$ ， $\lambda\in[0,1]$ ）也完全落在 $S$ 内。

> 补充（非原文）：为什么 $\lambda\mathbf{x}^{(a)}+(1-\lambda)\mathbf{x}^{(b)}$ 恰好表示连接 $\mathbf{x}^{(a)}$ 和 $\mathbf{x}^{(b)}$ 的直线段？把公式展开并重新提取公因式，可写成
>
> $$
> \mathbf{x}^{(b)}+\lambda\big(\mathbf{x}^{(a)}-\mathbf{x}^{(b)}\big)
> $$
>
> 在这个形式下几何意义很清晰：
>
> - $\mathbf{x}^{(b)}$ ：线段的起点位置。
> - $\mathbf{x}^{(a)}-\mathbf{x}^{(b)}$ ：从 $\mathbf{x}^{(b)}$ 指向 $\mathbf{x}^{(a)}$ 的方向向量。
> - $\lambda$ ：控制沿这个方向向量移动多远的比例参数。 $\lambda=0$ 时在起点 $\mathbf{x}^{(b)}$ ， $\lambda=1$ 时到达 $\mathbf{x}^{(a)}$ ， $\lambda\in[0,1]$ 扫过两点之间的整条线段。

#### 半空间是凸集 (a half-space is convex)

**题目**：证明如下定义的半空间 $H$ 是凸的，其中 $\mathbf{a}\in\mathbb{R}^d$ 、 $b\in\mathbb{R}$ ：

$$
H=\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}+b\geq0\}
$$

**证明**：任取两点 $\mathbf{x}^{(a)},\mathbf{x}^{(b)}\in H$ ，由 $H$ 的定义有

$$
\mathbf{a}^\top\mathbf{x}^{(a)}+b\geq0,\qquad\mathbf{a}^\top\mathbf{x}^{(b)}+b\geq0
$$

下面说明这两点的任意凸组合 (convex combination) 也落在这个半空间内。对任意 $\lambda\in[0,1]$ ，

$$
\begin{aligned}
&\mathbf{a}^\top\big(\lambda\mathbf{x}^{(a)}+(1-\lambda)\mathbf{x}^{(b)}\big)+b\\
&=\lambda\big(\mathbf{a}^\top\mathbf{x}^{(a)}+b\big)+(1-\lambda)\big(\mathbf{a}^\top\mathbf{x}^{(b)}+b\big)\geq0\\
&\Rightarrow\lambda\mathbf{x}^{(a)}+(1-\lambda)\mathbf{x}^{(b)}\in H
\end{aligned}
$$

> 补充（非原文）：等号成立是因为 $b=\lambda b+(1-\lambda)b$ ，可把 $b$ 拆进两个括号； $\geq0$ 成立是因为 $\lambda\geq0$ 、 $1-\lambda\geq0$ ，且两个括号都 $\geq0$ ，非负数的非负加权和仍非负。

## 二、定理与证明：XOR 问题线性不可分

**问题设定**：XOR（异或）函数在恰好一个输入为 1 时输出 1，否则输出 0，对应 4 个数据点 $(0,0)\to0$ 、 $(0,1)\to1$ 、 $(1,0)\to1$ 、 $(1,1)\to0$ 。**命题**：不存在权重 $(b,w_1,w_2)$ 能让线性分类器 $f(\mathbf{x})=b+w_1x_1+w_2x_2$ （配合阈值 0）正确分类全部 4 个点。

**证明一（代数视角：权重空间约束矛盾）**

对目标为 1 的点要求 $b+w_1x_1+w_2x_2\geq0$ ，目标为 0 的点要求 $b+w_1x_1+w_2x_2<0$ 。代入 4 个 XOR 点得到 4 条约束：

$$
b<0,\qquad b+w_2\geq0,\qquad b+w_1\geq0,\qquad b+w_1+w_2<0
$$

把两条"正例"约束相加： $2b+w_1+w_2\geq0$ ；把两条"负例"约束相加： $2b+w_1+w_2<0$ 。同一个量 $2b+w_1+w_2$ 不可能同时" $\geq0$ "又" $<0$ "，产生矛盾——说明满足全部 4 条约束的可行域为空，即不存在能正确分类所有点的权重。

**证明二（几何视角：特征空间中的凸性，Geometric Perspective: Convexity in Feature Space）**

记 XOR 的四个点为 $A=(0,1)$ 、 $C=(1,0)$ （正例，目标为 1）和 $B=(0,0)$ 、 $D=(1,1)$ （负例，目标为 0）， $F=(0.5,0.5)$ 。

**证明**：假设线性分类器

$$
f(\mathbf{x})=b+w_1x_1+w_2x_2
$$

正确分类了全部四个 XOR 点。那么正例落在半空间

$$
H^+=\{\mathbf{x}:f(\mathbf{x})\geq0\}
$$

中，负例落在半空间

$$
H^-=\{\mathbf{x}:f(\mathbf{x})<0\}
$$

中。两个半空间都是凸集（见第一节"半空间是凸集"）。

点 $F$ 是线段 $\overline{AC}$ 的中点，因为

$$
F=\frac{A+C}{2}=\frac{\begin{bmatrix}0\\1\end{bmatrix}+\begin{bmatrix}1\\0\end{bmatrix}}{2}=\begin{bmatrix}0.5\\0.5\end{bmatrix}
$$

点 $F$ 也是线段 $\overline{BD}$ 的中点，因为

$$
F=\frac{B+D}{2}=\frac{\begin{bmatrix}0\\0\end{bmatrix}+\begin{bmatrix}1\\1\end{bmatrix}}{2}=\begin{bmatrix}0.5\\0.5\end{bmatrix}
$$

- 因为 $A$ 和 $C$ 在 $H^+$ 中且 $H^+$ 是凸集，所以 $F$ 也在 $H^+$ 中。
- 因为 $B$ 和 $D$ 在 $H^-$ 中且 $H^-$ 是凸集，所以 $F$ 也在 $H^-$ 中。

但 $H^+$ 与 $H^-$ 不相交 (disjoint)， $F$ 不可能同时属于两者——矛盾。因此 XOR 数据不是线性可分的。 $\square$

> 补充（非原文）：中点就是 $\lambda=\tfrac12$ 的凸组合，所以能直接用凸集定义。 $H^-$ 用的是严格不等号 $<0$ ，第一节的证明写的是 $\geq0$ 的情形，但同样的推导对 $<0$ 也成立： $\lambda$ 与 $1-\lambda$ 非负且和为 1，两个负数的这种加权和仍为负。

两种证明用的是同一个核心思想（把"正确分类"翻译成对权重/空间区域的约束，再证明约束互相矛盾），只是分别在权重空间和特征空间中展开。

## 三、方法论：用特征工程缓解（但不能根治）线性局限

**思路**：在拟合线性边界之前，先对输入做非线性变换（加入多项式项/交互项），使数据在扩展后的特征空间中变得线性可分。

**XOR 的具体做法**：加入交互特征 $x_3=x_1x_2$ ，4 个点在 $(x_1,x_2,x_3)$ 空间中的约束变为一组可行的线性不等式组，例如取 $b=-0.5,w_1=1,w_2=1,w_3=-3$ 即可正确分类全部 4 点。对应分离超平面 $x_1+x_2-3x_1x_2=0.5$ 投影回原始 $(x_1,x_2)$ 空间时，是一条**曲线**——说明"整体仍是线性模型"和"能拟合非线性边界"并不矛盾，关键在于特征本身是否线性可分。

**这种方法的局限**：① 需要人工设计交互项，正确的组合往往不明显，依赖领域知识和反复试验；② 特征数量随阶数组合爆炸增长（ $d$ 个输入的二阶交互项就有 $O(d^2)$ 个），维度高时不可行；③ 若不加正则化地大量添加特征，容易过拟合。

## 四、核心概念与思想：动机指向神经网络

线性模型的这一结构性限制，恰好是引出**神经网络 (neural networks)** 的动机：与其手工设计特征映射，不如让模型直接从数据中**学习**非线性表示。可以这样理解神经网络的结构——它的最后一步仍然是一个线性模型，只是输入的特征 $\phi(\mathbf{x})$ 不再是手工设定的，而是由前面的隐藏层自动学出来的。

## 五、例子（精简保留）

- **XOR 扩展特征验证**：加入 $x_3=x_1x_2$ 后， $(0,0,0)\to f=-0.5<0$ ， $(0,1,0)\to f=0.5>0$ ， $(1,0,0)\to f=0.5>0$ ， $(1,1,1)\to f=-1.5<0$ ，四个点全部按目标值正确分类，验证了扩展特征空间确实线性可分。
