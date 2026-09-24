# Week 2 — Decision Trees

错题记录：

## 1. 互信息（信息增益）是否恒为非负

**题目**：Information gain about random variable $X$ by observing random variable $Y$ is always nonnegative. True or False?

**答案**：True

**解析**：

观察 $Y$ 带来的关于 $X$ 的信息增益就是**互信息 (mutual information)** $I(X;Y)$，定义为由于知道 $Y$ 而减少的关于 $X$ 的不确定性：

$$
I(X;Y)=H(X)-H(X\vert Y)
$$

互信息恒为非负，即 $I(X;Y)\geq0$——这就是信息论中的 **"信息不会产生负面作用" (information can't hurt)** 原则：多观察一个变量，最坏情况下也不会让不确定性变大。

$I(X;Y)=0$ 当且仅当 $X,Y$ 相互独立（此时观察 $Y$ 对 $X$ 的不确定性毫无帮助）。

这也是决策树用**信息增益**作为分裂准则的理论基础：选择某个特征分裂节点后，标签的条件熵 $H(t\vert\text{split})$ 不会比分裂前的熵 $H(t)$ 更大，即信息增益 $H(t)-H(t\vert\text{split})\geq0$ 恒成立，所以贪心地选信息增益最大的特征分裂总是"不亏"的。

## 2. 什么是决策树

> 来源：Practice Week 2 Lecture Prepare Quiz: Decision Trees，Question 1（第 1 次作答误选 B）

**题目**：What is a decision tree?

- A. A sequence of layered if-else statements used to make predictions on new data
- B. A method that always compares a test point to every training example in the dataset
- C. A model that can only be used for regression problems and not classification
- D. A random process for selecting class labels without using input features

**答案**：A

**解析**：

决策树 (decision tree) 本质上就是**一层套一层的 if-else 语句**：每个非叶节点检验一个特征上的条件（如 $x_j\leq s$ ），根据结果走向某个子节点；到达叶节点时输出该叶节点的预测值。预测新样本时只需沿一条从根到叶的路径走下去，不需要回看训练集。

- B 描述的是 **k 近邻 (kNN)**：kNN 是非参数、"懒惰"的方法，预测时要把测试点和**每一个**训练样本比较距离——这正是决策树与 kNN 的核心区别（决策树训练完就把训练数据"压缩"进了树结构里）。
- C 错：决策树既可做分类（叶节点输出多数类）也可做回归（叶节点输出均值）。
- D 错：决策树的每一次分裂都依赖输入特征。

## 3. 决策边界的定义

> 来源：Practice Week 2 Lecture Prepare Quiz: Decision Trees，Question 3（第 1 次作答误选 B）

**题目**：What is a decision boundary?

- A. The set of training examples that are misclassified
- B. A set of lines that separates data perfectly based on their class
- C. A partition of the data space corresponding to predicted classes
- D. The maximum depth of a decision tree

**答案**：C

**解析**：

**决策边界 (decision boundary)** 是由**模型的预测**决定的：它把输入空间划分成若干区域，每个区域内的所有点都被预测为同一个类别，区域之间的分界就是决策边界。

B 错在两点：

- 边界**不一定能完美分开数据**——决策边界反映的是模型"预测成什么"，而不是数据的真实类别；只要模型在训练集上有误差，就会有训练点落在"错误"的一侧。
- 边界**不一定是直线**——kNN 的边界可以是任意形状的折线/曲线；决策树的边界虽然由轴对齐的线段组成，但整体是分段的矩形区域，而不是"一组把数据完美分开的直线"。

参见 `decision_trees/dt_intro.md` 的"决策边界的轴对齐性"。

## 4. 决策树的决策边界是否总是轴对齐

> 来源：Practice Week 2 Lecture Prepare Quiz: Decision Trees，Question 11（第 1 次作答误选 False）

**题目**：Decision tree decision boundaries are always axis-aligned because each split tests a threshold on a single feature. True or False?

**答案**：True

**解析**：

决策树的每次分裂形如 $x_j\leq s$ ，其中 $x_j$ 是输入向量 $\mathbf{x}\in\mathbb{R}^D$ 的第 $j$ 个特征（ $D$ 为特征数）， $s$ 是阈值。这个条件在输入空间里对应的分界面是 $x_j=s$ ——一个**垂直于第 $j$ 个坐标轴**的超平面（二维时就是一条水平或竖直的线）。

由于每次分裂只看**一个**特征，没有任何分裂会产生形如 $w_1x_1+w_2x_2=c$ 的斜线，所以整棵树的决策边界只能由这些轴对齐的片段拼成，即**分段轴对齐的矩形区域**。

容易混淆的点：决策树是万能函数逼近器，足够深的树可以用很多小"台阶"**逼近**一条斜的边界——但逼近出来的边界本身仍然是轴对齐的阶梯形，而不是真正的斜线。这也是决策树的**归纳偏置 (inductive bias)**。参见 `decision_trees/dt_intro.md` 与 `decision_trees/dt_learn.md`。

## 5. 为什么连续特征只需考虑有限个分裂阈值

> 来源：Practice Week 2 Lecture Prepare Quiz: Decision Trees，Question 15（第 1 次作答误选 B）

**题目**：Why do decision tree algorithms only need to evaluate finitely many split thresholds for a continuous feature?

- A. Only thresholds between two observed values of the feature can change how the data points are partitioned
- B. Continuous features must first be converted into a small number of discrete categories before any splitting occurs
- C. Decision trees are limited to working with features that have a small and bounded range of values
- D. Every valid threshold must be chosen to exactly equal one of the feature values observed in the training data

**答案**：A

**解析**：

设训练集有 $N$ 个样本，某连续特征在训练集中的取值排序后为 $v_1<v_2<\dots<v_m$ （ $m\leq N$ 个不同取值）。阈值 $s$ 虽然可以取无穷多个实数，但只要两个阈值 $s,s'$ 落在同一个区间 $(v_i,v_{i+1})$ 内，它们对训练点的划分 $\{x_j\leq s\}$ / $\{x_j>s\}$ **完全相同**，因此信息增益也相同。

所以真正不同的划分最多只有 $m-1$ 种，取每对相邻取值的中点 $\frac{v_i+v_{i+1}}{2}$ 作为代表即可；取值范围之外的阈值不会产生两个非空子集，也没有意义。

- B 错：决策树**直接**在连续特征上做阈值分裂，不需要先离散化成少数几个类别。
- C 错：特征范围可以是任意的，有限性来自**训练集有限**，而不是特征范围有界。
- D 错：阈值通常取相邻两个取值的**中点**，并不要求恰好等于某个观测值。

参见 `decision_trees/dt_infogain.md` 方法论第 4 点、`decision_trees/dt_learn.md` 方法论第 2 点。

## 6. 决策树的停止准则（多选）

> 来源：Practice Week 2 Lecture Prepare Quiz: Decision Trees，Question 16（第 1 次作答漏选 C，得 0.67 分）

**题目**：Which of the following can be used as a stopping criterion during decision tree learning? Select all that apply.

- A. Stop splitting when the tree reaches a predefined maximum depth
- B. Stop splitting when a node contains fewer than a minimum number of samples
- C. Stop splitting when the best split produces too little information gain
- D. Stop splitting when all remaining features have different numeric scales

**答案**：A、B、C

**解析**：

**停止准则 (stopping criteria)** 都是控制树大小（即模型容量）的**超参数**，通常用验证集调优：

- A **最大深度 (maximum depth)**：树长到预设深度就停。
- B **最小分裂样本数 (minimum number of samples)**：节点样本太少时，再分裂得到的统计量不可靠，容易拟合噪声。
- C **最小信息增益 (minimum information gain)**：若当前节点的最优分裂带来的信息增益 $IG(Y\vert X)=H(Y)-H(Y\vert X)$ 低于阈值，说明这次分裂几乎不能降低标签的不确定性，就拒绝分裂、把该节点设为叶节点——漏选的就是这一项。
- D 错：决策树每次分裂只比较**单个特征**与阈值，对特征的数值尺度不敏感（不像 kNN 或梯度下降那样需要标准化），尺度不同与是否停止分裂无关。

此外，节点已经**纯 (pure)**（所有样本标签相同）时也显然应停止。参见 `decision_trees/dt_learn.md` 方法论第 3 点。
