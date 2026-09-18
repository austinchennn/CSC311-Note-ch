> 来源: https://www.teach.cs.toronto.edu/~csc311notes/linear_classification/softmax_regression.html

# Multi-Class Classification (Softmax Regression) — 笔记

## 一、核心定义

**多分类问题 (multi-class classification)**：每个样本恰好属于 $K(>2)$ 个类别之一的分类问题，是二分类的推广（$K=2$ 时退化为二分类）。

**独热向量 (one-hot vector)**：$K$ 分类的标签用 $\mathbb{R}^K$ 中的向量表示，正确类别对应位置为 1、其余为 0。之所以不直接用整数编码类别（如 Oak=1, Maple=2, Birch=3），是因为整数编码会人为引入类别间不存在的"顺序"和"距离"关系（例如让算法误以为类别 1 比类别 3 更接近类别 2），而独热编码对所有类别一视同仁。

**logits（未归一化得分）**：对每个类别 $k$ 单独用一个线性函数打分 $z_k$，所有类别的打分堆叠成向量 $\mathbf{z}\in\mathbb{R}^K$；这些原始分数还不能直接解释为概率。

**权重矩阵 (weight matrix)** $\mathbf{W}\in\mathbb{R}^{K\times(D+1)}$：第 $k$ 行是计算第 $k$ 个类别得分所需的权重（含偏置）。

**Softmax 函数**：把 logits 向量转换成一个合法的概率分布（各分量非负且和为 1）的函数；是 sigmoid 从二分类到多分类的推广。

**多分类交叉熵损失 (multi-class cross-entropy loss)**：二分类交叉熵在 $K$ 类情形下的推广，用独热目标向量与预测概率向量的对数做内积取负。

## 二、公式

每个类别的线性得分与堆叠成的得分向量：

$$
z_k=\sum_{j=1}^D w_{k,j}x_j+b_k=\mathbf{w}_k^\top\mathbf{x},\qquad \mathbf{z}=\mathbf{W}\mathbf{x}
$$

Softmax 函数：

$$
y_k=\frac{\exp(z_k)}{\sum_{k'=1}^K\exp(z_{k'})}
$$

多分类交叉熵损失（单样本，向量形式）：

$$
\mathcal{L}_{CE}=-\sum_{k=1}^K t_k\log(y_k)=-\mathbf{t}^\top\log(\mathbf{y})
$$

由于 $\mathbf{t}$ 是独热向量，上式只剩一项非零——损失恰好等于**真实类别对应概率的负对数**，即模型给正确类别分配的概率越低，损失越大。

梯度与更新规则：

$$
\frac{\partial\mathcal{L}_{CE}}{\partial\mathbf{w}_k}=(y_k-t_k)\mathbf{x},\qquad
\mathbf{w}_k\leftarrow\mathbf{w}_k-\frac{\alpha}{N}\sum_{i=1}^N(y_k^{(i)}-t_k^{(i)})\mathbf{x}^{(i)}
$$

与二分类逻辑回归的梯度形式 $(y-t)\mathbf{x}$ 完全一致，只是现在对每个类别 $k$ 各自维护一套权重 $\mathbf{w}_k$。

## 三、核心概念与思想

- **决策边界仍是超平面**：类别 $i,j$ 之间的边界是 $z_i=z_j$，即 $(\mathbf{w}_i-\mathbf{w}_j)^\top\mathbf{x}=0$——仍然是一个超平面。$K$ 个类别两两之间共有 $K(K-1)/2$ 条这样的边界，共同把输入空间划分成 $K$ 个凸区域，边界的交集是"所有类别得分相等"的最大模糊区。
- **softmax 不改变决策边界**：预测类别只取决于哪个 $z_k$ 最大，而 softmax 保持各 logit 的相对大小顺序不变，因此 softmax 只影响"概率/置信度"的解读方式，完全不影响分类结果和边界位置——推断阶段甚至可以跳过 softmax，直接取 $\arg\max_k z_k$。
- **softmax 是 argmax 的光滑近似**：argmax 是硬选择、不可导；softmax 根据相对得分平滑地分配概率，因此在训练阶段（需要梯度）必须使用 softmax，即使推断阶段可以省略。
- **与二分类逻辑回归的统一性**：无论是激活函数（sigmoid→softmax）还是损失函数（二元交叉熵→多分类交叉熵），还是最终的梯度形式，多分类的推广都与二分类保持高度一致的数学结构。

## 四、例子（精简保留）

- **三分类叶片数据集**：把叶片分类扩展为 Oak/Maple/Birch 三类，一个 Maple 样本（数据集中第 5 个样本）的独热目标向量为 $\mathbf{t}^{(5)}=[0,1,0]^\top$。
