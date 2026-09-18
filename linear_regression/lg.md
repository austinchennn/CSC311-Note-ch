> 来源: https://www.teach.cs.toronto.edu/~csc311notes/linear_regression/lg.html

# Linear Regression — 笔记

## 一、核心定义

**归纳偏置 (inductive bias)**：模型对"输入-输出关系"所做的假设；由没有免费午餐定理 (no free lunch theorem) 可知，不存在普适最优算法，因此选对归纳偏置（即选对模型族）是学习能否成功的关键。

**线性模型 (linear model，标量形式)**：限制映射函数 $f$ 为特征的线性组合：

$$
y=f(\mathbf{x})=w_1x_1+w_2x_2+\dots+w_Dx_D+b=\sum_{j=1}^D w_jx_j+b
$$

其中 **权重向量 (weight vector)** $\mathbf{w}$ 控制各特征的贡献大小/方向，**偏置 (bias / intercept)** $b$ 是让超平面 (hyperplane) 可以不过原点的平移项——若去掉 $b$ ，模型被迫过原点，表达能力大打折扣，因此偏置项通常必须保留。

**向量化 (vectorization)**：把标量形式的模型改写成矩阵/向量运算，目的不仅是数学上简洁，更是为了利用 GPU/CPU 的并行计算能力和 NumPy/PyTorch 等库的高度优化实现（避免逐元素 Python 循环的开销）。

**技巧：把偏置当作权重**：引入恒为 1 的哑特征 (dummy feature) $x_0=1$ ，对应权重 $w_0=b$ ，即可把偏置吸收进权重向量，模型统一写成 $y=\mathbf{w}^\top\mathbf{x}$ 。

**数据矩阵 / 设计矩阵 (data matrix / design matrix)** $\mathbf{X}\in\mathbb{R}^{N\times(D+1)}$ ：每一行是一个样本的（含哑特征的）特征向量。

**模型参数 (model parameter) vs 超参数 (hyperparameter)**：前者由学习算法从训练数据中自动学出（如 $\mathbf{w}$ ），后者是训练前人为设定、通常用验证集调优的设置（如 kNN 的 $K$ ）。

**损失函数 (loss function)**：衡量单个样本预测值与真实值之间差异的函数。回归常用**平方误差损失 (squared error loss)**：

$$
\mathcal{L}(y,t)=\frac{1}{2}(y-t)^2
$$

差值 $y-t$ 称为**残差 (residual)**；乘 $1/2$ 是为了求导时抵消平方产生的系数 2。

**代价函数 (cost function)**：训练集上所有样本损失的平均值，本质上是关于参数 $\mathbf{w}$ 的函数：

$$
\mathcal{E}(\mathbf{w})=\frac{1}{N}\sum_{i=1}^N\mathcal{L}(y^{(i)},t^{(i)})
$$

代入平方误差得到**均方误差 (mean squared error, MSE)** 代价函数。

**$L^2$ 范数 (Euclidean norm)**： $\lVert\mathbf{v}\rVert_2=\sqrt{\sum_i v_i^2}$ ，满足 $\lVert\mathbf{v}\rVert_2^2=\mathbf{v}^\top\mathbf{v}$ 。

## 二、公式（向量化结果）

预测（整份训练集）：

$$
\mathbf{y}=\mathbf{X}\mathbf{w}
$$

MSE 代价函数的等价写法：

$$
\mathcal{E}(\mathbf{w})=\frac{1}{2N}\sum_{i=1}^N(y^{(i)}-t^{(i)})^2=\frac{1}{2N}(\mathbf{X}\mathbf{w}-\mathbf{t})^\top(\mathbf{X}\mathbf{w}-\mathbf{t})=\frac{1}{2N}\lVert\mathbf{X}\mathbf{w}-\mathbf{t}\rVert_2^2
$$

代价函数的梯度（向量化）：

$$
\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\frac{1}{N}\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{t})
$$

直接解（闭式解，closed-form solution）：

$$
\mathbf{w}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{t}
$$

## 三、推导过程

**单样本梯度（标量链式法则）**：损失对权重 $w_j$ 求导，用链式法则拆成 $\frac{\partial\mathcal{L}}{\partial y^{(i)}}=(y^{(i)}-t^{(i)})$ 与 $\frac{\partial y^{(i)}}{\partial w_j}=x_j^{(i)}$ 两部分相乘，得到

$$
\frac{\partial\mathcal{L}(\mathbf{w})}{\partial w_j}=(y^{(i)}-t^{(i)})x_j^{(i)}\;\Longrightarrow\;\nabla_{\mathbf{w}}\mathcal{L}(\mathbf{w})=(\mathbf{w}^\top\mathbf{x}^{(i)}-t^{(i)})\mathbf{x}^{(i)}
$$

对所有样本取平均，即得代价函数的（未向量化）梯度 $\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\frac{1}{N}\sum_i(\mathbf{w}^\top\mathbf{x}^{(i)}-t^{(i)})\mathbf{x}^{(i)}$ 。

**三步向量化梯度**：
1. 定义残差向量 $\mathbf{r}\in\mathbb{R}^{N\times1}$ ， $r^{(i)}=\mathbf{w}^\top\mathbf{x}^{(i)}-t^{(i)}$ ；根据矩阵乘法维度匹配 + "结果需为列向量"的约定， $\sum_i r^{(i)}\mathbf{x}^{(i)}=\mathbf{X}^\top\mathbf{r}$ （而非 $\mathbf{r}^\top\mathbf{X}$ ，后者是行向量）。
2. 把残差写成预测向量减目标向量： $\mathbf{r}=\mathbf{y}-\mathbf{t}$ 。
3. 把预测向量写成 $\mathbf{y}=\mathbf{X}\mathbf{w}$ ，代入即得最终结果 $\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\frac{1}{N}\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{t})$ 。

**求直接解（令梯度为零）**：

$$
\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\mathbf{0}\;\Rightarrow\;\mathbf{X}^\top\mathbf{X}\mathbf{w}=\mathbf{X}^\top\mathbf{t}\;\Rightarrow\;\mathbf{w}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{t}
$$

注意 $\mathbf{X}^\top\mathbf{X}$ 是矩阵，不能"相除"，需两边左乘其逆矩阵。

## 四、方法论：求最优参数的两种途径

1. **直接解 (direct solution)**：① 求代价函数对每个参数的偏导（梯度）；② 令梯度为零；③ 解方程组得到参数——一步到位的闭式解。
2. **梯度下降 (gradient descent)**：迭代式方法（下一篇笔记详述）。

**直接解的优缺点**：
- 优点：平方损失是凸函数 (convex)，只有唯一全局最小值，因此闭式解存在且唯一，无需迭代。
- 缺点：① 仅适用于少数问题（闭式解存在的情形很少）；② 需要计算矩阵求逆 $(\mathbf{X}^\top\mathbf{X})^{-1}$ ，复杂度 $O(D^3)$ ，特征多时代价高；③ 无法推广到本课程后续大多数复杂模型。

## 五、核心概念与思想：模块化机器学习 (modular approach)

任何一个机器学习模型都可拆解成三个可独立替换的模块：
1. **模型架构 (model architecture)**：输入到输出的映射形式（如特征的线性组合）；
2. **损失/代价函数 (loss/cost function)**：如何量化预测误差（如平方误差）；
3. **优化算法 (optimization algorithm)**：如何调整参数以降低损失（如直接解、梯度下降）。

这种模块化思维呼应 **Idea #1：学习即优化 (Learning is Optimization)**——把"选一个好的预测器"问题转化为"在一族模型上最小化目标函数"问题；固定其中两个模块、替换第三个，就能派生出许多不同的机器学习算法。

## 六、例子（精简保留）

- **回归任务示例**：预测房价（特征：面积/位置/房间数）、预测降雨量（特征：温度/湿度/风速）——目标变量均为连续值，区别于分类任务的离散类别。
- **偏置项必要性的一维例子**： $y=wx+b$ 中若去掉 $b$ ，直线被迫过原点 $(0,0)$ ，无法拟合不过原点的数据。
