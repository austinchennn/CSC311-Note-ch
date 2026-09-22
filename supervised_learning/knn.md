> 来源: https://www.teach.cs.toronto.edu/~csc311notes/supervised_learning/knn.html

# k-Nearest Neighbours — 笔记

## 一、核心定义

**1-最近邻分类器 (1-NN)**

对测试点 $\mathbf{x}$ ，在训练集中找到距离最近的点 $\mathbf{x}^{(C)}$ ：

$$
C=\underset{i\in\{1,\dots,N\}}{\arg\min}\ \text{distance}(\mathbf{x}^{(i)},\mathbf{x})
$$

然后直接复制其标签作为预测： $y=t^{(C)}$ 。**没有真正的"训练"过程**（惰性学习 / lazy learning，全部计算延迟到推断阶段）。

**k-最近邻算法 (k-NN)**

找到距 $\mathbf{x}$ 最近的 $k$ 个训练样本，取其标签的**多数类 (majority class / majority vote)** 作为预测。

**Voronoi 图 (Voronoi diagram)**

把空间划分为若干区域，每个区域内的点都离某一训练样本最近；1-NN 的决策边界即 Voronoi 图中不同类别区域的分界线（由线段构成）。

**超参数 (hyperparameter)** $k$

需要人工/验证集选定、不由训练过程自动学习的设置。

**过拟合 (overfitting)**

模型学到训练数据中不可泛化的噪声/个别样本特性（ $k$ 过小，如 $k=1$ 时训练准确率恒为100%，但边界"锯齿状"、对噪声敏感）。

**欠拟合 (underfitting)**

模型过于简单、无法捕捉真实规律（ $k$ 过大，极端情况 $k=N$ 时永远预测训练集中的多数类）。

**标准化 / 归一化 (standardize / normalize)**

将每个特征变换为均值0、方差1，避免量纲不同导致某特征主导距离计算。

**维度灾难 (curse of dimensionality)**

高维空间中大多数点彼此距离都很远，"最近邻"的概念因而变得不再有意义。

## 二、公式

**距离度量**：
- 欧氏距离 (Euclidean distance, $L_2$ )： $\displaystyle \|\mathbf{x}^{(a)}-\mathbf{x}^{(b)}\|_2=\sqrt{\sum_{j=1}^d (x_j^{(a)}-x_j^{(b)})^2}$ （默认选择）
- 余弦相似度 (cosine similarity)： $\displaystyle \text{cosine}(\mathbf{x}^{(a)},\mathbf{x}^{(b)})=\frac{\mathbf{x}^{(a)}\cdot\mathbf{x}^{(b)}}{\|\mathbf{x}^{(a)}\|_2\|\mathbf{x}^{(b)}\|_2}$ （更关注方向而非幅度，适合文本分类等）
- 一般 $L_p$ 距离 ( $L_p$ distance)： $\displaystyle \|\mathbf{x}^{(a)}-\mathbf{x}^{(b)}\|_p=\Big(\sum_{j=1}^d |x_j^{(a)}-x_j^{(b)}|^p\Big)^{1/p}$
- 切比雪夫距离 (Chebyshev distance, $L_\infty$ )： $\displaystyle \|\mathbf{x}^{(a)}-\mathbf{x}^{(b)}\|_\infty=\max_j |x_j^{(a)}-x_j^{(b)}|$

**标准化**：用训练集算出每个特征的均值 $\mu_j$ 、标准差 $\sigma_j$ ，令 $\tilde x_j=\dfrac{x_j-\mu_j}{\sigma_j}$ ；验证/测试集必须复用训练集算出的 $\mu_j,\sigma_j$ ，不能单独计算（防止数据泄漏）。

**覆盖单位超立方体所需训练点数下界**：

$$
N\geq\left(\frac{1}{2\epsilon}\right)^d
$$

当 $\epsilon<\tfrac12$ 时该下界随维度 $d$ **指数增长**。

## 三、推导过程（维度灾难）

设训练数据均匀分布在单位超立方体 $[0,1]^d$ 中，用切比雪夫距离定义"接近原点"：距原点 $\epsilon$ 以内的区域是边长为 $\epsilon$ 的小立方体 $[0,\epsilon]^d$ ，体积为 $\epsilon^d$ ；由于整个超立方体体积为1，落在该区域内的点的比例即为 $\epsilon^d$ ——随维度 $d$ 增大而指数级缩小。

进一步：若要让**每个**训练点覆盖半径 $\epsilon$ 内的区域（体积 $\leq(2\epsilon)^d$ ）， $N$ 个训练点总覆盖体积至多为 $N(2\epsilon)^d$ ，要覆盖整个体积为1的超立方体，需满足

$$
N(2\epsilon)^d\geq 1 \;\Longrightarrow\; N\geq\left(\frac{1}{2\epsilon}\right)^d,
$$

即所需训练点数随维度指数增长。

## 四、方法论

- **选择 $k$**：理论上最优 $k$ 与样本量 $n$ 有关，需满足 $k\to\infty$ 且 $k/n\to0$ （经验法则 $k<\sqrt n$ ）；实践中直接在**验证集 (validation set)** 上尝试不同 $k$ 值，选表现最好的。
- **特征标准化流程**：① 仅用训练集计算每个特征的 $\mu_j,\sigma_j$ ；② 用同一组统计量变换训练/验证/测试集的对应特征。
- kNN 对特征所用的单位/量纲高度敏感：量纲不一致时某特征会主导距离计算，因此标准化是默认操作。

## 五、核心概念与思想

- **懒惰学习 (lazy learning)**：kNN 无显式训练阶段，全部计算（存储训练集、算距离）都发生在推断时，训练集越大，存储与推断代价越高。
- **模型复杂度 (model complexity) 与偏差-方差权衡 (bias-variance tradeoff)**（呼应 Idea #2）： $k$ 直接控制模型复杂度—— $k$ 减小 → 更能拟合精细模式但更易过拟合； $k$ 增大 → 预测更稳定但可能欠拟合。
- **几何视角**（呼应 Idea #4）：kNN 的表现完全依赖"距离"这一几何概念是否有意义，因此特征选择与特征缩放至关重要。
- **维度灾难的两个表现**：① 高维空间中"接近原点"的比例随维度指数级缩小，覆盖空间所需样本数量随维度指数增长；② 高维空间中各点到某一测试点的距离趋于相近，"最近邻"不再有区分度。
- kNN 虽简单且有维度灾难等局限，但仍是重要的基线方法，也常作为更复杂算法的组件（在自动学习出的特征空间上做距离计算）；数据足够密集时，很多复杂方法的行为会趋近于最近邻方法。

## 六、例子（精简保留）

- **叶片数据集**：测试点 (12.0, 16.0)——1-NN 最近训练点为 $\mathbf{x}^{(3)}=(10,18)$ （Oak），预测 Oak；3-NN 决策边界比 1-NN 更平滑，对单个训练点的敏感度更低。
- **量纲敏感性示例**：把叶宽单位从 cm 改成 mm 后，测试点在 1-NN 下的最近邻从 $\mathbf{x}^{(3)}$ 变成 $\mathbf{x}^{(5)}$ ，预测结果随之改变——说明未标准化时量纲会主导距离计算。
