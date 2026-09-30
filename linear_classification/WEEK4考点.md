> 来源: 用户提供的课件内容（特征映射、正则化、二元线性分类、逻辑回归与梯度下降变体考点梳理，非 csc311notes 网站原文）

# Week 4 考点梳理 — 从特征映射到逻辑回归

本章把线性回归的两个收尾话题（特征映射、正则化）与线性分类/逻辑回归串成一条线：先解决"模型太弱/太强"的复杂度问题，再把输出从实数换成类别，最后讨论如何高效地做梯度下降。

## 一、核心定义

#### 特征映射 (feature mapping)

把原始输入 $x$ 映射到新特征空间 $\psi(x)$ ，再在 $\psi(x)$ 上做线性回归，从而用"对参数线性"的模型拟合对输入非线性的关系。多项式特征映射取 $\psi(x)=[1,x,x^2,\dots,x^M]^\top$ ，其中 $M$ 为**多项式阶数 (polynomial degree)**，控制模型复杂度。

#### 欠拟合 / 过拟合 (underfitting / overfitting)

- **欠拟合**：模型拟合能力不足，训练误差和测试误差都高（如 $M$ 太小）。
- **过拟合**：模型完美拟合训练数据但泛化差，训练误差低、测试误差高（如 $M=9$ 时）。此时权重量级急剧增大，函数在数据点之间剧烈震荡。

#### $L^2$ 正则化 / 岭回归 ($L^2$ regularization / ridge regression)

在保持参数数量较大的同时，通过惩罚权重大小强制模型选择更"简单"的解：

$$
\mathcal{E}_{reg}(\mathbf{w})=\mathcal{E}(\mathbf{w})+\frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2
$$

其中 $\mathcal{E}(\mathbf{w})$ 是原始代价（如平方误差）， $\lambda\ge0$ 是**正则化系数 (regularization hyperparameter)**，在数据拟合度与权重大小之间做权衡。

#### 二元线性分类 (binary linear classification)

目标 $t\in\{0,1\}$ 。借助**虚拟特征 (dummy feature)** $x_0=1$ 把偏置吸收进权重，得到 $z=\mathbf{w}^\top\mathbf{x}$ ，其中 $\mathbf{x}\in\mathbb{R}^{D+1}$ 为扩展输入、 $\mathbf{w}\in\mathbb{R}^{D+1}$ 为权重（ $D$ 为原始特征数）。预测规则：

$$
y=\begin{cases}1 & z\ge0\\0 & z<0\end{cases}
$$

#### 决策边界 (decision boundary)

满足 $\mathbf{w}^\top\mathbf{x}=0$ 的超平面，把输入空间分成预测为 1 和预测为 0 的两半。

#### 线性可分 (linearly separable)

存在某个 $\mathbf{w}$ 能把所有训练样本都分对；此时可以把每个样本写成一条关于 $\mathbf{w}$ 的不等式，联立求解。

## 二、公式

#### 逻辑回归完整架构 (logistic regression)

1. **线性映射**： $z=\mathbf{w}^\top\mathbf{x}$
2. **Sigmoid 激活 (sigmoid / logistic function)**：

$$
y=\sigma(z)=\frac{1}{1+e^{-z}}\in(0,1)
$$

3. **交叉熵损失 (cross-entropy loss)**：

$$
\mathcal{L}_{CE}(y,t)=-t\log y-(1-t)\log(1-y)
$$

#### 四种损失函数的演进

| 损失 | 形式 | 问题 |
| --- | --- | --- |
| 0-1 损失 (0-1 loss) | $\mathbb{I}[y\ne t]$ | 在 $z=0$ 处梯度未定义，其余处处为 0，无学习信号 |
| 平方损失 (squared loss) | $\frac{1}{2}(z-t)^2$ | $z$ 无界，极端但**正确**的预测（如 $t=1,z=10$ ）反而受巨大惩罚 |
| Sigmoid + 平方损失 | $\frac{1}{2}(\sigma(z)-t)^2$ | 极端**错误**预测时梯度 $\approx0$ （梯度消失），参数卡在临界点 |
| Sigmoid + 交叉熵 | $\mathcal{L}_{CE}(\sigma(z),t)$ | 预测严重偏离时仍提供强梯度 ✓ |

## 三、推导过程

#### 为什么 Sigmoid + 平方损失会梯度消失 (vanishing gradient)

设 $y=\sigma(z)$ ， $\mathcal{L}=\frac{1}{2}(y-t)^2$ 。链式法则：

$$
\frac{\partial\mathcal{L}}{\partial w_j}=\frac{\partial\mathcal{L}}{\partial y}\cdot\frac{\partial y}{\partial z}\cdot\frac{\partial z}{\partial w_j}=(y-t)\cdot\sigma(z)\big(1-\sigma(z)\big)\cdot x_j
$$

若 $t=1$ 但 $z\to-\infty$ （极端错误）， $\sigma(z)\to0$ ，于是 $\sigma(z)(1-\sigma(z))\to0$ ——即使误差 $(y-t)\approx-1$ 很大，整体梯度仍 $\approx0$ 。

#### 交叉熵如何消除这一影响

$$
\frac{\partial\mathcal{L}_{CE}}{\partial y}=-\frac{t}{y}+\frac{1-t}{1-y}
$$

与 $\sigma'(z)=y(1-y)$ 相乘后分母被约掉：

$$
\frac{\partial\mathcal{L}_{CE}}{\partial z}=y-t,\qquad\frac{\partial\mathcal{L}_{CE}}{\partial w_j}=(y-t)\,x_j
$$

对数的求导产生 $\frac{1}{y}$ ，恰好抵消 sigmoid 饱和带来的小因子，所以严重错误时梯度大小 $\approx|x_j|$ ，信号依然强。

## 四、方法论

1. **判断过拟合/欠拟合**：
   - 随 $M$ 增大，训练误差**单调递减**；测试误差**先降后升**，最低点就是欠拟合→过拟合的临界点。
   - 训练、测试误差都高 → 欠拟合；训练低、测试高 → 过拟合。

2. **调 $\lambda$ 对权重的影响**：
   - $\lambda$ 过大 → 所有权重被压到极小，模型欠拟合。
   - $\lambda$ 过小 → 无法限制权重，模型过拟合。

3. **纠正方向**：
   - 过拟合 → 降低 $M$ 或增大 $\lambda$ 。
   - 欠拟合 → 提高 $M$ 或减小 $\lambda$ 。

4. **手推线性分类权重（不等式法）**：
   - 对每个样本 $(\mathbf{x}^{(i)},t^{(i)})$ ：若 $t^{(i)}=1$ 写 $\mathbf{w}^\top\mathbf{x}^{(i)}\ge0$ ；若 $t^{(i)}=0$ 写 $\mathbf{w}^\top\mathbf{x}^{(i)}<0$ 。
   - 联立不等式，挑一组满足全部约束的 $[w_0,w_1,w_2]$ ，再画出 $w_0+w_1x_1+w_2x_2=0$ 。

5. **梯度下降变体选择**：
   - **批量梯度下降 (batch GD)**：每步遍历全部 $N$ 个样本；梯度准确、可完全向量化，但更新慢、计算贵。
   - **随机梯度下降 (stochastic GD)**：每步只用 1 个随机样本；更新极快，但方差大、噪声高、无法向量化。
   - **小批量梯度下降 (mini-batch GD)**：每步用大小为 $k$ 的小批次；批次大 → 更新慢但准确、利于向量化；批次小 → 更新快但噪声大、难向量化。

## 五、核心概念与思想

- **复杂度靠两个旋钮控制**：阶数 $M$ 决定"能表示多复杂"，$\lambda$ 决定"愿意多复杂"。
- **损失函数要与激活函数匹配**：sigmoid 的饱和区需要一个带对数的损失来抵消，这是交叉熵的设计动机。
- **梯度下降的三维权衡**：计算成本、梯度噪声、向量化支持。

## 六、例子（精简保留）

- **逻辑与 (AND)**：数据 $(0,0)\to0$ 、 $(0,1)\to0$ 、 $(1,0)\to0$ 、 $(1,1)\to1$ 。约束为

$$
w_0<0,\quad w_0+w_2<0,\quad w_0+w_1<0,\quad w_0+w_1+w_2\ge0
$$

  一个可行解： $w_0=-1.5,\ w_1=1,\ w_2=1$ ，决策边界 $x_1+x_2=1.5$ 。
- **$M=9$ 多项式拟合**：训练误差趋近 0，但权重量级暴涨、曲线在点间剧烈震荡——典型过拟合。

## 七、必会知识点

- 识别模型复杂度对训练/测试误差的影响曲线，准确判定过拟合与欠拟合的临界点。
- 掌握带 $L^2$ 惩罚项的代价函数结构，以及 $\lambda$ 增大/减小对权重缩放的影响。
- 给定若干坐标点及 0/1 标签，快速写出 $\mathbf{w}^\top\mathbf{x}\ge0$ 或 $<0$ 的约束并推出决策边界。
- 熟记逻辑回归的线性映射、sigmoid、二元交叉熵三个公式。
- 理解 "Sigmoid + 平方损失" 在极大误差时导数 $\approx0$ 的原因，以及交叉熵如何通过对数消除这一影响。

## 八、潜在考点预测

1. **数学推导与证明题**：给一组二维数据点（类似 Logical AND），构建线性不等式组，手推能完美分离数据的 $[w_0,w_1,w_2]$ ，并在坐标系画出 $\mathbf{w}^\top\mathbf{x}=0$ 。
2. **损失函数缺陷分析题**：给一个不匹配的激活+损失组合（如直接对线性分类用平方损失），求对 $\mathbf{w}$ 的偏导，并结合公式分析极端错误预测时梯度消失的根本原因。
3. **超参数调优与模型诊断**：给出拟合散点图或误差曲线，诊断过拟合/欠拟合，并说明应如何调整 $M$ 或 $\lambda$ 。
4. **优化器选择与权衡**：给定场景（如流式数据、内存受限硬件），在 Batch GD / SGD / Mini-Batch GD 中选择，并从计算成本、梯度噪声、向量化支持三个维度论证。
