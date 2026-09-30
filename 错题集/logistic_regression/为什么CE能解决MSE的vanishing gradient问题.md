# 为什么 CE 能解决 MSE 的梯度消失问题 (vanishing gradient)

**题目**：设一个正样本真实标签 $t=1$ ，模型的线性部分给出极端错误的 $z=-1000$ 。为什么用 Sigmoid + 平方损失 (squared error, SE) 时模型几乎学不动，而换成交叉熵 (cross-entropy, CE) 后能强烈修正？

**答案**：Sigmoid 的导数 $y(1-y)$ 在饱和区趋于 0。平方损失的梯度 $(y-t)\,y(1-y)$ 会被这个小因子压到 $\approx0$ ；交叉熵对 $y$ 的导数 $\frac{y-t}{y(1-y)}$ 恰好把它约掉，得到 $\frac{\partial L_{CE}}{\partial z}=y-t\approx-1$ ，梯度仍然很大。

**解析**：

#### 设定

线性映射 $z=\mathbf{w}^\top\mathbf{x}$ ，Sigmoid 激活

$$
y=\sigma(z)=\frac{1}{1+e^{-z}}
$$

$z=-1000$ 时 $y\approx0$ ，而真实标签 $t=1$ ——这是一个"非常错" (very wrong) 的预测。

#### 问题根源：Sigmoid 的导数衰减

$$
\frac{\partial y}{\partial z}=y(1-y)
$$

当 $y$ 接近 0 或 1 时， $y(1-y)\approx0$ 。本例中 $y(1-y)\approx0\times1=0$ ：预测越离谱，激活函数的导数反而越接近 0。

#### 为什么平方损失会梯度消失

平方损失 $L_{SE}=\frac{1}{2}(y-t)^2$ ，按链式法则：

$$
\frac{\partial L_{SE}}{\partial z}=\frac{\partial L_{SE}}{\partial y}\cdot\frac{\partial y}{\partial z}=(y-t)\cdot y(1-y)
$$

代入 $y\approx0,\ t=1$ ：

$$
\frac{\partial L_{SE}}{\partial z}\approx(0-1)\times0\approx0
$$

即使预测非常错，梯度依然 $\approx0$ 。模型会误以为已经到达临界点 (critical point)，停止更新参数，无法纠正这个巨大错误。

#### 交叉熵如何解决

交叉熵损失：

$$
L_{CE}=-t\log y-(1-t)\log(1-y)
$$

对 $y$ 求偏导：

$$
\frac{\partial L_{CE}}{\partial y}=-\frac{t}{y}+\frac{1-t}{1-y}=\frac{y-t}{y(1-y)}
$$

再乘以 Sigmoid 的导数，分子分母完美抵消：

$$
\frac{\partial L_{CE}}{\partial z}=\frac{y-t}{y(1-y)}\cdot y(1-y)=y-t
$$

代入 $y\approx0,\ t=1$ ：

$$
\frac{\partial L_{CE}}{\partial z}=0-1=-1
$$

梯度不再是 0，而是直接等于误差 $(y-t)$ 。对数项求导产生的 $\frac{1}{y(1-y)}$ 抵消了 Sigmoid 饱和带来的小因子，所以在 $y$ 严重偏离 $t$ 时仍有很大的梯度 (large gradient) 信号，迫使模型强烈修正。对权重则有 $\frac{\partial L_{CE}}{\partial w_j}=(y-t)\,x_j$ 。

> 相关笔记：[linear_classification/logistic_regression.md](../../linear_classification/logistic_regression.md)、[linear_classification/WEEK4考点.md](../../linear_classification/WEEK4考点.md)
