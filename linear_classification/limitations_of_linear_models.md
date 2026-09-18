> 来源: https://www.teach.cs.toronto.edu/~csc311notes/linear_classification/limitations_of_linear_models.html

# Limitations of Linear Models — 笔记

## 一、核心定义

**线性可分 (linearly separable)**：两组点存在一个能把它们完全分开的超平面。线性回归、逻辑回归、softmax 回归共享同一个结构性限制——预测都基于输入特征的线性函数，因此分类边界永远只能是超平面；当真实类别边界是曲线或由多个不连通区域组成时，无论用多少训练数据、用什么优化方法，线性模型都无法完美分类。

**半空间 (half-space)**：$\mathbb{R}^d$ 中形如 $H=\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}+b\geq0\}$ 的点集，即超平面 $\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}+b=0\}$ 一侧的所有点。

**凸集 (convex set)**：集合 $S$ 中任取两点 $\mathbf{x}^{(a)},\mathbf{x}^{(b)}$，连接它们的线段（即所有 $\lambda\mathbf{x}^{(a)}+(1-\lambda)\mathbf{x}^{(b)}$，$\lambda\in[0,1]$）也完全落在 $S$ 内。
