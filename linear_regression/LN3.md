> 来源: 用户提供的课件内容（线性回归 Linear Regression 与梯度下降 Gradient Descent 综述，非 csc311notes 网站原文）

# Linear Regression & Gradient Descent — 课件笔记 (LN3)

## 一、核心定义

#### 模型 (model)

用于描述输入变量与输出变量之间关系的函数形式。

#### 损失/代价函数 (loss/cost function)

用于量化模型对数据的拟合程度有多差；损失函数针对单个样本，代价函数是整个训练集上损失函数的平均值。

#### 优化算法 (optimization algorithm)

用于拟合出一个能够使代价函数最小化的模型参数的算法。

#### 线性模型 (linear model)

$$
y=w_1x_1+\dots+w_Dx_D+b
$$

其中 $\mathbf{x}=(x_1,\dots,x_D)\in\mathbb{R}^D$ 是输入特征向量， $D$ 是特征维度； $\mathbf{w}=(w_1,\dots,w_D)\in\mathbb{R}^D$ 是权重向量； $b\in\mathbb{R}$ 是偏置项； $y\in\mathbb{R}$ 是标量预测目标。

#### 虚拟特征 / 哑特征 (dummy feature)

在输入向量中添加一个恒为 1 的特征 $x_0=1$ ，从而把偏置 $b$ 吸收进权重向量，把模型简化为

$$
y=\mathbf{w}^\top\mathbf{x}
$$

此时 $\mathbf{w},\mathbf{x}\in\mathbb{R}^{D+1}$ （即 $w_0=b$ ， $x_0=1$ ）。

#### 向量化 (vectorization)

把针对单个样本的标量运算，改写成对整份训练集的矩阵运算 $\mathbf{y}=\mathbf{X}\mathbf{w}$ ，其中设计矩阵 $\mathbf{X}\in\mathbb{R}^{N\times(D+1)}$ （每行是一个样本含哑特征的特征向量， $N$ 是样本数）， $\mathbf{y}\in\mathbb{R}^N$ 是对整份训练集的预测值向量。

- **向量化的优势**：利用 CPU/GPU 的并行计算提升速度；无需显式循环，代码更简洁；更有效地使用内存；能利用 NumPy 等库的底层优化。

#### 损失函数：平方误差损失 (squared error loss)

$$
L=\frac{1}{2}(y^{(i)}-t^{(i)})^2
$$

其中 $y^{(i)}$ 是第 $i$ 个样本的预测值， $t^{(i)}$ 是第 $i$ 个样本的真实目标值。用平方形式方便求导，用系数 $\frac{1}{2}$ 抵消求导产生的 2；平方还能方便地最小化残差 $y^{(i)}-t^{(i)}$ 的幅度。

#### 代价函数：均方误差 (mean squared error, MSE)，向量化形式

$$
\mathcal{E}(\mathbf{w})=\frac{1}{2N}\lVert\mathbf{X}\mathbf{w}-\mathbf{t}\rVert_2^2
$$

其中 $\mathbf{t}\in\mathbb{R}^N$ 是整份训练集的真实目标向量。

#### 直接求解法 (direct solution)

对代价函数求关于权重 $\mathbf{w}$ 的导数，令导数等于零，直接解方程组得到最优参数的方法——一步到位，不需要迭代。

#### 梯度下降 (gradient descent)

一种通用的迭代式优化方法：从随机初始点 $\mathbf{w}_0$ 出发，重复应用更新规则，直到满足停止条件为止。

#### 学习率 (learning rate) $\alpha$

控制梯度下降每一步更新幅度大小的超参数。

## 二、公式

代价函数的梯度：

$$
\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\frac{1}{N}\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{t})
$$

令梯度为零得到的直接解（闭式解）：

$$
\mathbf{w}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{t}
$$

梯度下降的通用更新规则：

$$
\mathbf{w}\leftarrow\mathbf{w}-\alpha\nabla_{\mathbf{w}}F(\mathbf{w})
$$

代入线性回归的梯度后，线性回归专用的梯度下降更新规则：

$$
\mathbf{w}\leftarrow\mathbf{w}-\frac{\alpha}{N}\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{t})
$$

## 三、推导过程

**直接解的推导思路**：对代价函数 $\mathcal{E}(\mathbf{w})$ 求关于 $\mathbf{w}$ 的梯度，令梯度 $\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\mathbf{0}$ ，解出的 $\mathbf{w}$ 即为使代价函数最小的参数，得到 $\mathbf{w}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{t}$ 。

**梯度下降更新方向与步长的确定**：
- 更新方向应与梯度的符号相反（即朝下降方向走），因为梯度指向的是使函数值增大最快的方向。
- 更新步长应与梯度的大小成正比——曲线陡峭（梯度大）时迈大步，曲线平缓（梯度小）时迈小步；这样离最小值远时走得快、离最小值近时走得慢。

## 四、方法论

1. **直接求解法 vs 梯度下降法的选择**：
   - 直接求解法：一步到位，不需要迭代；但当特征维度 $D$ 很大时，矩阵求逆 $(\mathbf{X}^\top\mathbf{X})^{-1}$ 的计算成本非常高昂（时间复杂度 $O(D^3)$ ），且无法推广到其他模型或其他损失函数。
   - 梯度下降法：通用的函数优化方法，比直接求解更容易实现；对高维数据（大 $D$ ）更高效，因为每次更新的复杂度仅为 $O(ND)$ 。
2. **停止条件的选择**：
   - 理论上：在参数 $\mathbf{w}$ 不再变化时停止。
   - 实践中：通常在代价函数的变化足够小时停止，或迭代次数耗尽（不想再等了）时停止。
3. **学习率 $\alpha$ 的选择**：
   - $\alpha$ 太小 → 模型收敛需要的时间太长。
   - $\alpha$ 太大 → 模型可能发散 (diverge)，无法找到最小值。

## 五、核心概念与思想

- **机器学习的模块化视角 (a modular approach to ML)**：机器学习问题通常可拆解成三个相互独立的模块——模型 (model)、损失/代价函数 (loss/cost function)、优化算法 (optimization algorithm)；固定其中两个模块、替换第三个，就能派生出不同的机器学习算法。呼应 [[lg]] 中"模块化机器学习"与 Idea #1（学习即优化）。
- **直接解法的局限性**：直接求解法虽然对线性回归+平方损失有效，但这一特殊性无法推广——一旦换成其他模型或其他损失函数，往往求不出闭式解，因此梯度下降作为通用方法更具普适性。

## 六、例子（精简保留）

- **回归目标示例**：通过面积和位置预测房价；通过历史销售数据预测公司收入——目标变量均为连续的标量值。
