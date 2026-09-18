> 来源: https://www.teach.cs.toronto.edu/~csc311notes/linear_classification/limitations_of_linear_models.html

# Limitations of Linear Models — 笔记

## 一、核心定义

**线性可分 (linearly separable)**：两组点存在一个能把它们完全分开的超平面。线性回归、逻辑回归、softmax 回归共享同一个结构性限制——预测都基于输入特征的线性函数，因此分类边界永远只能是超平面；当真实类别边界是曲线或由多个不连通区域组成时，无论用多少训练数据、用什么优化方法，线性模型都无法完美分类。

**半空间 (half-space)**：$\mathbb{R}^d$ 中形如 $H=\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}+b\geq0\}$ 的点集，即超平面 $\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}+b=0\}$ 一侧的所有点。

**凸集 (convex set)**：集合 $S$ 中任取两点 $\mathbf{x}^{(a)},\mathbf{x}^{(b)}$，连接它们的线段（即所有 $\lambda\mathbf{x}^{(a)}+(1-\lambda)\mathbf{x}^{(b)}$，$\lambda\in[0,1]$）也完全落在 $S$ 内。

## 二、定理与证明：XOR 问题线性不可分

**问题设定**：XOR（异或）函数在恰好一个输入为 1 时输出 1，否则输出 0，对应 4 个数据点 $(0,0)\to0$、$(0,1)\to1$、$(1,0)\to1$、$(1,1)\to0$。**命题**：不存在权重 $(b,w_1,w_2)$ 能让线性分类器 $f(\mathbf{x})=b+w_1x_1+w_2x_2$（配合阈值 0）正确分类全部 4 个点。

**证明一（代数视角：权重空间约束矛盾）**

对目标为 1 的点要求 $b+w_1x_1+w_2x_2\geq0$，目标为 0 的点要求 $b+w_1x_1+w_2x_2<0$。代入 4 个 XOR 点得到 4 条约束：

$$
b<0,\qquad b+w_2\geq0,\qquad b+w_1\geq0,\qquad b+w_1+w_2<0
$$

把两条"正例"约束相加：$2b+w_1+w_2\geq0$；把两条"负例"约束相加：$2b+w_1+w_2<0$。同一个量 $2b+w_1+w_2$ 不可能同时"$\geq0$"又"$<0$"，产生矛盾——说明满足全部 4 条约束的可行域为空，即不存在能正确分类所有点的权重。

**证明二（几何视角：凸性 + 中点矛盾）**

先证明半空间是凸集：设 $H=\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}+b\geq0\}$，任取 $\mathbf{x}^{(a)},\mathbf{x}^{(b)}\in H$、$\lambda\in[0,1]$，则

$$
\mathbf{a}^\top\big(\lambda\mathbf{x}^{(a)}+(1-\lambda)\mathbf{x}^{(b)}\big)+b=\lambda(\mathbf{a}^\top\mathbf{x}^{(a)}+b)+(1-\lambda)(\mathbf{a}^\top\mathbf{x}^{(b)}+b)\geq0
$$

（两项都非负、系数也非负，故和非负），所以线段上的点仍在 $H$ 内，半空间是凸集。

若线性分类器 $f$ 能正确分开 XOR 的两类，则正例都落在开半空间 $H^+=\{\mathbf{x}:f(\mathbf{x})>0\}$，负例都落在 $H^-=\{\mathbf{x}:f(\mathbf{x})<0\}$，且由上述结论 $H^+,H^-$ 都是凸集。

关键观察：两个正例 $(0,1),(1,0)$ 的中点，与两个负例 $(0,0),(1,1)$ 的中点，是**同一个点** $(0.5,0.5)$。由 $H^+$ 的凸性，该中点必须落在 $H^+$ 中（因为它是两个正例的中点）；由 $H^-$ 的凸性，该中点又必须落在 $H^-$ 中（因为它也是两个负例的中点）。但 $H^+$ 与 $H^-$ 不相交，同一个点 $f(0.5,0.5)$ 不可能同时为正又为负——矛盾。故 XOR 不是线性可分的。$\blacksquare$

两种证明用的是同一个核心思想（把"正确分类"翻译成对权重/空间区域的约束，再证明约束互相矛盾），只是分别在权重空间和特征空间中展开。
