> 来源: https://www.teach.cs.toronto.edu/~csc311notes/linear_classification/logistic_regression.html

# Logistic Regression — 笔记

## 一、核心定义

#### 正例 / 负例 (positive / negative examples)

约定 $t=1$ 的样本为正例， $t=0$ 的样本为负例。

#### 预激活值 / logit (pre-activation / logit)

线性打分 $z=\mathbf{w}^\top\mathbf{x}$ ，是送入激活函数之前的原始值；之所以叫 logit，是因为若把 $y=\sigma(z)$ 解释为概率，则 $z=\log\big(\frac{y}{1-y}\big)$ 恰是对数几率 (log-odds)。

#### 决策边界 / 可行域 / 线性可分 (decision boundary / feasible region / linearly separable)

分类边界发生在 $z=0$ 处，是输入空间中的一个超平面；每个训练样本会在**权重空间**中划出一个半空间约束，所有约束的交集称为**可行域 (feasible region)**——若可行域非空，则训练集**线性可分 (linearly separable)**。

#### 零一损失 / 分类误差 (zero-one loss / classification error)

预测对得 0、错得 1 的损失，精确对应我们关心的"分类是否正确"，但几乎处处梯度为零，无法用于梯度下降训练。

#### Sigmoid（逻辑）函数 (sigmoid / logistic function)

把任意实数压缩到 $(0,1)$ 区间、可解释为"属于正类的概率"的平滑函数，是 logistic regression 用来替代硬阈值的关键设计。

#### 交叉熵损失 (cross-entropy loss)

专为配合 sigmoid 设计的损失函数，能在模型自信但预测错误时仍提供强梯度信号；其负对数似然的形式使得最小化交叉熵等价于**最大似然估计 (maximum likelihood estimation)**。

#### 逻辑回归模型 (logistic regression model)

由 sigmoid 激活 + 交叉熵损失组成的完整二分类模型；参数是权重 $\mathbf{w}$ ，通过训练集上交叉熵损失的平均值（即代价函数/目标函数, cost/objective function）来学习。

#### 参数化模型 vs 非参数化模型 (parametric vs non-parametric model)

假设空间能否用固定数量的参数完全描述——逻辑回归是参数化的（只需存 $D$ 个权重），k近邻是非参数化的（需要存整个训练集，复杂度随数据量增长）。

#### 归纳偏置 (inductive bias)

学习算法用来从训练数据泛化到新样本的假设/偏好集合；逻辑回归的归纳偏置是"类别可用线性边界分开"，kNN 的归纳偏置是"相近的输入应有相近的输出"。

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

**为什么不能用平方误差**：对一个被正确分类且高置信度的正例（如 $z=5$ ），平方误差 $(z-t)^2$ 仍然很大（如 $(5-1)^2=16$ ），会错误地惩罚"正确且自信"的预测，把决策边界推向错误方向。

**为什么不能用零一损失**：由链式法则 $\frac{\partial\mathcal L_{0-1}}{\partial w_j}=\frac{\partial\mathcal L_{0-1}}{\partial z}\frac{\partial z}{\partial w_j}$ ，而 $\mathcal L_{0-1}$ 只在 $z$ 跨过阈值的瞬间才改变，其余位置导数恒为零——梯度下降完全无法更新参数。

**为什么 sigmoid + 平方误差也不行**：设正例 $t=1$ 被严重误分类， $z=-5\Rightarrow y=\sigma(-5)\approx0.0067$ 。梯度中含有因子 $y(1-y)\approx0.0067\times0.9933\approx0.0066$ ，几乎为零——即使模型错得离谱，权重更新量也微乎其微。这说明 sigmoid 的"饱和区"恰好会把误差信号压缩没了，模型组件之间并非完全独立、可以随意搭配。

这就是**梯度消失 (vanishing gradient)**，具体说是由**激活函数饱和 (saturation)** 引起的梯度消失。由链式法则，平方误差配 sigmoid 的梯度为

$$
\frac{\partial\mathcal{L}_{SE}}{\partial w_j}=(y-t)\cdot y(1-y)\cdot x_j
$$

其中 $y(1-y)$ 正是 sigmoid 的导数（见下方推导）。当 $|z|$ 很大时， $y$ 极其接近 $0$ 或 $1$ ，落在 sigmoid 曲线两端的平坦区， $y(1-y)\approx0$ ；梯度是各因子连乘，这一个近零因子就把整个梯度拉到 $0$ ，哪怕误差项 $(y-t)$ 已经大到极限。

*例（ $z=\pm1000$ ）*：第一行 $t=1$ 、 $z=-1000$ ，有 $y=\sigma(-1000)\approx0$ ，预测错得离谱，但

$$
\frac{\partial\mathcal{L}_{SE}}{\partial w_j}=\underbrace{(y-t)}_{\approx-1}\cdot\underbrace{y(1-y)}_{\approx0}\cdot x_j\approx0
$$

第一个因子 $(y-t)$ 已经取到最大绝对值，如实反映"离目标很远"；第二个因子却把梯度压成 $0$ 。第二行同理： $y=\sigma(1000)\approx1$ ， $(y-t)\approx1$ ，但 $y(1-y)$ 再次趋近 $0$ 。

| 目标 $t$ | $z$ | 预测 $y$ | 对/错 | 梯度 | 看起来是临界点？ |
|:-:|:-:|:-:|:-:|:-:|:-:|
| $1$ | $-1000$ | $\approx0$ | 错得离谱 | $\approx0$ | 是 |
| $0$ | $1000$ | $\approx1$ | 错得离谱 | $\approx0$ | 是 |

**结论**：预测**错得离谱**时，权重反而看起来**处在临界点 (critical point)**（梯度 $\approx0$ ）。梯度下降依赖梯度提供学习信号——模型最需要调整权重，更新量却微乎其微，**恰好卡死在最需要学习的地方**。这也再次说明平方误差不适合做分类损失，引出下面的交叉熵。

**sigmoid 导数的推导**：把 sigmoid 写成 $\sigma(z)=(1+e^{-z})^{-1}$ ，用链式法则得

$$
\frac{d\sigma(z)}{dz}=-(1+e^{-z})^{-2}\cdot\frac{d}{dz}(1+e^{-z})=-(1+e^{-z})^{-2}\cdot(-e^{-z})=\frac{e^{-z}}{(1+e^{-z})^2}
$$

这个结果是对的，但还不是想要的形式。关键一步是**把它拆成两个因子**，第一个因子就是 $\sigma(z)$ ；第二个因子用"分子加一减一"的技巧改写成 $1-\sigma(z)$ ：

$$
\frac{e^{-z}}{(1+e^{-z})^2}=\frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}},\qquad
\frac{e^{-z}}{1+e^{-z}}=\frac{(1+e^{-z})-1}{1+e^{-z}}=1-\frac{1}{1+e^{-z}}=1-\sigma(z)
$$

两者相乘即得

$$
\frac{d\sigma(z)}{dz}=\sigma(z)(1-\sigma(z))
$$

——导数完全由函数值本身表示，前向传播算出 $y=\sigma(z)$ 后，反向传播直接用 $y(1-y)$ 即可，无需再算指数。下面交叉熵梯度推导中的因子 $\frac{dy}{dz}=y(1-y)$ 就来自这里。

**交叉熵梯度的推导（关键消去）**：交叉熵的主要优势在求梯度时才显现出来。sigmoid 激活与交叉熵损失组合成的完整模型是

$$
\begin{aligned}
&\text{线性模型：} && z=\mathbf{w}^\top\mathbf{x}\\
&\text{计算预测：} && y=\sigma(z)=\frac{1}{1+e^{-z}}\\
&\text{计算损失：} && \mathcal{L}_{CE}(y,t)=-t\log(y)-(1-t)\log(1-y)
\end{aligned}
$$

权重 $w_j$ 先影响 $z$ ，再通过 $y$ 影响损失，因此用链式法则拆成三项：

$$
\frac{\partial\mathcal{L}_{CE}}{\partial w_j}=\frac{\partial\mathcal{L}_{CE}}{\partial y}\cdot\frac{dy}{dz}\cdot\frac{\partial z}{\partial w_j}
$$

逐项计算（第一项通分： $-t(1-y)+(1-t)y=-t+ty+y-ty=y-t$ ；第二项即上面推导的 sigmoid 导数；第三项因 $z=\sum_j w_jx_j$ ）：

$$
\begin{aligned}
\frac{\partial\mathcal{L}_{CE}}{\partial y}&=-\frac{t}{y}+\frac{1-t}{1-y}=\frac{y-t}{y(1-y)}\\
\frac{dy}{dz}&=y(1-y)\\
\frac{\partial z}{\partial w_j}&=x_j
\end{aligned}
$$

三项相乘，第一项的分母 $y(1-y)$ 恰好与第二项 $y(1-y)$ **完全抵消**：

$$
\frac{\partial\mathcal{L}_{CE}}{\partial w_j}=\frac{y-t}{y(1-y)}\cdot y(1-y)\cdot x_j=(y-t)x_j
$$

——这正是"sigmoid 饱和消失"问题被交叉熵损失解决的数学原因：无论 $y$ 离 $t$ 多远，梯度大小都直接正比于误差 $(y-t)$ ，不会因为 sigmoid 的平坦区而消失。并且这个梯度形式与线性回归平方误差的梯度形式完全一致，这不是巧合——线性回归与逻辑回归同属**广义线性模型 (generalized linear model)** 家族（详见 [generalized_linear_model.md](generalized_linear_model.md)）。

## 四、方法论与核心思想

- **失败驱动设计 (design by elimination)**：本章的方法论是"先试几种直观方案、分析它们为什么失败，再据此设计出正确方案"——平方误差失败在于惩罚了正确预测，零一损失失败在于梯度恒零，sigmoid+平方误差失败在于饱和区梯度消失；每一次失败都指向下一步该修正什么。
- **模块并非完全独立**：模型架构（sigmoid）、损失函数（交叉熵/平方误差）、优化器（梯度下降）虽然是可拆卸的模块，但彼此的组合方式会相互影响梯度行为，不能脱离数学细节随意搭配。
- **决策边界不变性**：无论用不用 sigmoid，只要阈值取在 $\sigma(z)=0.5$ （等价于 $z=0$ ），决策边界始终是同一个超平面 $\mathbf{w}^\top\mathbf{x}=0$ ——sigmoid 影响的是"概率解读"，不影响边界位置。
- **特征缩放**：与线性回归一样，逻辑回归的梯度下降也对特征尺度敏感，训练前通常做标准化 $\tilde{x}_j=\frac{x_j-\mu_j}{\sigma_j}$ 。
- **参数化 vs 非参数化的权衡**：逻辑回归计算快、只需存 $D$ 个权重、可解释性强，但受限于"线性边界"假设，边界复杂时会欠拟合；kNN 灵活、无需假设边界形状，但预测慢、需存全部训练集、高维时受维度灾难影响。两者没有绝对优劣，选择取决于数据量、计算资源与归纳偏置是否匹配真实模式。

## 五、例子（精简保留）

- **叶片决策边界数值例**：权重 $(w_0,w_1,w_2)=(-150,5,10)$ 对应决策边界 $x_2=15-0.5x_1$ ；用叶 #1 $(x_1,x_2)=(7,12)$ 代入得 $z=5\not<0$ ，说明这组权重会把该 Oak 叶误分类为 Maple——直观展示"每个样本在权重空间中划出一个约束半空间"。
- **sigmoid+平方误差梯度消失的数值例**： $t=1$ 、 $z=-5$ 时 $y\approx0.0067$ ，梯度因子 $y(1-y)\approx0.0066$ ，几乎不会更新参数，即使预测严重错误。
