> 来源: 补充内容（非课程原文）；相关课程笔记见 [logistic_regression.md](logistic_regression.md)

# Generalized Linear Model (GLM) — 笔记

**一句话**：广义线性模型 (generalized linear model, GLM) 保留"线性"二字，是因为它的**核心计算引擎仍是特征的线性组合** $z=\mathbf{w}^\top\mathbf{x}+b$ ；sigmoid 等非线性函数只是套在这个线性组合外面的"转换器"。

## 一、为什么仍叫"线性"：三个角度

### 1. 对参数 $\mathbf{w}$ 线性

- 说一个模型"线性"，通常指**线性预测器 (linear predictor)** $z$ 对**待学习的参数**是线性的。
- 逻辑回归中，不论输入特征 $\mathbf{x}$ 事先怎么变换，它和权重的结合方式永远是点乘相加 $z=w_1x_1+w_2x_2+\dots+b$ 。
- sigmoid 只是拿到 $z$ 之后的一步后处理 $y=\sigma(z)$ ——知识（权重）仍以线性组合的形式存在。

### 2. 决策边界仍是平直的超平面

- 输出 $y$ 关于 $z$ 是 S 型曲线，但在特征空间中划出的分类界线是笔直的。
- 预测 $y=0.5$ （正负类各半）时，由 $\sigma(z)=0.5\iff z=0$ ，边界为

$$
\mathbf{w}^\top\mathbf{x}+b=0
$$

  ——一个标准的超平面。这就是逻辑回归被称为**线性分类器 (linear classifier)** 的原因。

### 3. "广义"的来源：联系函数 (link function)

- 普通线性回归默认目标服从正态分布、输出范围是 $(-\infty,+\infty)$ ；目标若是 0/1（伯努利分布）或非负整数计数（泊松分布），直接用它就不合适。
- GLM 保留线性内核 $z$ ，再用一个函数把 $z$ 映射到目标所需的取值范围。严格地说，**联系函数 $g$ 把均值映射到 $z$** （ $g(\mu)=z$ ），而把 $z$ 映射回预测值的是它的**逆** $g^{-1}$ ：

| 模型 | 目标分布 | 联系函数 $g(\mu)=z$ | 预测 $y=g^{-1}(z)$ | 输出范围 |
|:-:|:-:|:-:|:-:|:-:|
| 线性回归 | 正态 | 恒等 $\mu$ | $z$ | $(-\infty,+\infty)$ |
| 逻辑回归 | 伯努利 | logit $\log\frac{\mu}{1-\mu}$ | sigmoid $\frac{1}{1+e^{-z}}$ | $(0,1)$ |
| 泊松回归 | 泊松 | $\log\mu$ | $e^z$ | $(0,+\infty)$ |

  （所以常说的"逻辑回归的联系函数是 sigmoid"不严谨：sigmoid 是 logit 的逆函数。这也对应 [logistic_regression.md](logistic_regression.md) 里 $z=\log\frac{y}{1-y}$ 叫 logit 的原因。）

## 二、梯度形式一致不是巧合：正则联系函数 (canonical link function)

- 线性回归（平方误差）与逻辑回归（交叉熵）的梯度都是

$$
\frac{\partial\mathcal{L}}{\partial w_j}=(y-t)\,x_j\qquad\text{即 (预测 − 目标) × 输入}
$$

- 原因：只要为目标分布搭配它专属的**正则联系函数**（正态 ↔ 恒等，伯努利 ↔ logit，泊松 ↔ log），并用该分布的**负对数似然**作损失（正态 → 平方误差，伯努利 → 交叉熵），求导时非线性的导数项就恰好与损失导数的分母抵消。
- 逻辑回归中具体就是： $\frac{\partial\mathcal{L}_{CE}}{\partial y}=\frac{y-t}{y(1-y)}$ 的分母与 $\frac{dy}{dz}=y(1-y)$ 抵消（推导见 [logistic_regression.md](logistic_regression.md) 「三、推导过程」）。
- 反例：sigmoid + 平方误差**不是**正则搭配，没有这种抵消，梯度里残留 $y(1-y)$ ，于是在饱和区出现梯度消失。

## 三、总结

- **线性**：线性预测器 $z=\mathbf{w}^\top\mathbf{x}+b$ 对参数线性，且分类边界 $z=0$ 是超平面。
- **广义**：通过联系函数 $g$ （预测时用 $g^{-1}$ ）把 $z$ 接到不同的目标分布上——恒等 → 线性回归，logit/sigmoid → 逻辑回归，log/exp → 泊松回归。
- **梯度统一**：正则联系函数 + 负对数似然损失 ⇒ 梯度总是 $(y-t)\mathbf{x}$ 。
