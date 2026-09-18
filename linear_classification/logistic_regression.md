> 来源: https://www.teach.cs.toronto.edu/~csc311notes/linear_classification/logistic_regression.html

# Logistic Regression — 笔记

## 一、核心定义

**正例 / 负例 (positive / negative examples)**：约定 $t=1$ 的样本为正例，$t=0$ 的样本为负例。

**预激活值 / logit (pre-activation / logit)**：线性打分 $z=\mathbf{w}^\top\mathbf{x}$，是送入激活函数之前的原始值；之所以叫 logit，是因为若把 $y=\sigma(z)$ 解释为概率，则 $z=\log\big(\frac{y}{1-y}\big)$ 恰是对数几率 (log-odds)。

**决策边界 / 可行域 / 线性可分 (decision boundary / feasible region / linearly separable)**：分类边界发生在 $z=0$ 处，是输入空间中的一个超平面；每个训练样本会在**权重空间**中划出一个半空间约束，所有约束的交集称为**可行域 (feasible region)**——若可行域非空，则训练集**线性可分 (linearly separable)**。

**零一损失 / 分类误差 (zero-one loss / classification error)**：预测对得 0、错得 1 的损失，精确对应我们关心的"分类是否正确"，但几乎处处梯度为零，无法用于梯度下降训练。

**Sigmoid（逻辑）函数 (sigmoid / logistic function)**：把任意实数压缩到 $(0,1)$ 区间、可解释为"属于正类的概率"的平滑函数，是 logistic regression 用来替代硬阈值的关键设计。

**交叉熵损失 (cross-entropy loss)**：专为配合 sigmoid 设计的损失函数，能在模型自信但预测错误时仍提供强梯度信号；其负对数似然的形式使得最小化交叉熵等价于**最大似然估计 (maximum likelihood estimation)**。

**逻辑回归模型 (logistic regression model)**：由 sigmoid 激活 + 交叉熵损失组成的完整二分类模型；参数是权重 $\mathbf{w}$，通过训练集上交叉熵损失的平均值（即代价函数/目标函数, cost/objective function）来学习。

**参数化模型 vs 非参数化模型 (parametric vs non-parametric model)**：假设空间能否用固定数量的参数完全描述——逻辑回归是参数化的（只需存 $D$ 个权重），k近邻是非参数化的（需要存整个训练集，复杂度随数据量增长）。

**归纳偏置 (inductive bias)**：学习算法用来从训练数据泛化到新样本的假设/偏好集合；逻辑回归的归纳偏置是"类别可用线性边界分开"，kNN 的归纳偏置是"相近的输入应有相近的输出"。

## 二、公式

Sigmoid 函数及其导数：

$$
\sigma(z)=\frac{1}{1+e^{-z}},\qquad \sigma(-z)=1-\sigma(z),\qquad \frac{d\sigma}{dz}=\sigma(z)(1-\sigma(z))
$$

逻辑回归模型：

$$
y=\sigma(\mathbf{w}^\top\mathbf{x})=\frac{1}{1+e^{-\mathbf{w}^\top\mathbf{x}}}
$$

交叉熵损失（单样本）：

$$
\mathcal{L}_{CE}(y,t)=-t\log(y)-(1-t)\log(1-y)
$$

代价函数（训练集平均）：

$$
\mathcal{E}(\mathbf{w})=-\frac{1}{N}\sum_{i=1}^N\Big[t^{(i)}\log(y^{(i)})+(1-t^{(i)})\log(1-y^{(i)})\Big]
$$

梯度（关键结果）与梯度下降更新：

$$
\nabla_{\mathbf{w}}\mathcal{L}_{CE}=(y-t)\mathbf{x},\qquad
\mathbf{w}\leftarrow\mathbf{w}-\frac{\alpha}{N}\sum_{i=1}^N(y^{(i)}-t^{(i)})\mathbf{x}^{(i)}
$$

## 三、推导过程

**为什么不能用平方误差**：对一个被正确分类且高置信度的正例（如 $z=5$），平方误差 $(z-t)^2$ 仍然很大（如 $(5-1)^2=16$），会错误地惩罚"正确且自信"的预测，把决策边界推向错误方向。

**为什么不能用零一损失**：由链式法则 $\frac{\partial\mathcal{L}_{0-1}}{\partial w_j}=\frac{\partial\mathcal{L}_{0-1}}{\partial z}\frac{\partial z}{\partial w_j}$，而 $\mathcal{L}_{0-1}$ 只在 $z$ 跨过阈值的瞬间才改变，其余位置导数恒为零——梯度下降完全无法更新参数。

**为什么 sigmoid + 平方误差也不行**：设正例 $t=1$ 被严重误分类，$z=-5\Rightarrow y=\sigma(-5)\approx0.0067$。梯度中含有因子 $y(1-y)\approx0.0067\times0.9933\approx0.0066$，几乎为零——即使模型错得离谱，权重更新量也微乎其微。这说明 sigmoid 的"饱和区"恰好会把误差信号压缩没了，模型组件之间并非完全独立、可以随意搭配。

**交叉熵梯度的推导（关键消去）**：用链式法则 $\frac{\partial\mathcal{L}_{CE}}{\partial w_j}=\frac{\partial\mathcal{L}_{CE}}{\partial y}\cdot\frac{dy}{dz}\cdot\frac{\partial z}{\partial w_j}$，三项分别为 $\big(-\frac{t}{y}+\frac{1-t}{1-y}\big)$、$y(1-y)$、$x_j$。相乘后 $y(1-y)$ 恰好与 $\big(-\frac{t}{y}+\frac{1-t}{1-y}\big)$ 中的分母完全抵消，化简得到极简形式：

$$
\frac{\partial\mathcal{L}_{CE}}{\partial w_j}=(y-t)x_j
$$

——这正是"sigmoid 饱和消失"问题被交叉熵损失解决的数学原因：无论 $y$ 离 $t$ 多远，梯度大小都直接正比于误差 $(y-t)$，不会因为 sigmoid 的平坦区而消失。并且这个梯度形式与线性回归平方误差的梯度形式完全一致，这不是巧合——线性回归与逻辑回归同属**广义线性模型 (generalized linear model)** 家族。
