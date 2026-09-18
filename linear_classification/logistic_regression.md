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
