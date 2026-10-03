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

## 7. 决策树的参数与超参数

**题目**：What are some examples of model parameters of the decision tree model? What are some examples of hyperparameters of the decision tree model?

**答案**：参数是学习得到的：每个内部节点用哪个特征分裂、分裂阈值、叶节点的预测值。超参数由人设定：最大深度、最小分裂样本数、分裂准则（如 Gini 指数或熵）等。

**解析**：

**模型参数 (model parameters)** 由学习算法从训练数据中确定——对决策树而言就是树的结构本身：每个内部节点测试的特征 $x_j$ 与阈值 $t$（形如 $x_j \le t$），以及每个叶节点输出的类别（分类取多数类）或数值（回归取均值）。

**超参数 (hyperparameters)** 在训练前设定、控制学习过程与模型容量，通常用验证集调优：最大深度 (maximum depth)、最小分裂样本数 (minimum samples to split)、最小信息增益阈值、分裂准则 (splitting criterion，熵/信息增益或 Gini)。参见第 6 题的停止准则。

## 8. 同一特征能否在一条分支上被多次测试

**题目**：Can we test the same feature more than once along one branch in a decision tree? Justify your answer.

**答案**：可以。对连续特征，可以在不同深度用不同阈值反复测试同一特征。

**解析**：

例如根节点测试 $x_1 \le 5$，其左子节点再测试 $x_1 \le 2$，两者合起来把 $x_1$ 切成区间 $(-\infty,2]$、$(2,5]$。这正是决策树用多段轴对齐边界逼近复杂边界的方式（参见第 4 题）。

注意：对**离散特征**若按全部取值多路分裂，分过一次后该分支内该特征取值已固定，再测试没有信息增益，所以通常不会重复；但二元分裂（如 $x_j = a$ vs. $x_j \neq a$）时仍可再测。

## 9. "决策树是万能函数逼近器"的含义

**题目**："A decision tree is a universal function approximator." What does this statement mean?

**答案**：若不限制深度，决策树可以把每个训练样本分到自己的叶节点，从而完美拟合任意训练集（前提是不存在输入相同但标签冲突的样本）。

**解析**：

**万能函数逼近器 (universal function approximator)**：模型族足够灵活，能以任意精度表示（逼近）任意目标函数。对决策树：每条根到叶的路径对应输入空间里的一个轴对齐矩形区域，叶子足够多时，这些小矩形可以把任意函数逼近成分段常数函数；在训练集上可做到训练误差为 0。

代价是：这种无限容量意味着极易**过拟合 (overfitting)**，因此需要用第 6、7 题的超参数限制树的大小。

## 10. 决策树学习算法为什么是"贪心"的

**题目**：Our decision tree learning algorithm is a "greedy" algorithm. Explain what "greedy" means.

**答案**：在每个节点只选择当前局部最优的分裂（如信息增益最大的那个），不向前看这一选择是否能导向全局最优的树。

**解析**：

**贪心 (greedy)**：每一步做当下看起来最好的选择，且选定后不回溯。决策树在每个节点计算所有候选分裂的 $IG(Y\vert X)=H(Y)-H(Y\vert X)$，取最大者，然后递归处理子节点。

原因：寻找全局最优（如最小的、训练误差为零的）决策树是 NP 难问题，无法枚举所有树结构。代价：可能错过"当前增益小、但为后续分裂铺路"的分裂——经典例子是 XOR：单独看 $x_1$ 或 $x_2$ 的信息增益都为 0，但两者组合才能完美分类。

## 11. 节点处是否可能没有可用的特征/分裂

**题目**：Is it possible to run into a situation where there are no features/splits left to test at a node? Justify your answer briefly.

**答案**：可能。当节点上多个样本的所有特征取值完全相同、但标签不同时（通常源于噪声或矛盾数据），任何分裂都无法把它们分开。

**解析**：

此时任何候选分裂都会把这些样本分到同一个子节点，信息增益为 0；对离散特征，沿分支所有特征也可能都已被用完。节点既不纯又无法继续分裂，只能停止并设为叶节点，预测取多数类（或输出类别比例作为概率）。这也说明第 9 题"完美拟合训练集"必须有"无冲突标签"的前提。

## 12. 深树 vs. 浅树的优缺点

**题目**：What are the advantages/disadvantages of a tree that is deep (versus shallow)? Justify your answer. You may want to consider underfitting/overfitting, computational cost of training/making predictions, and the interpretability of the model.

**答案**：

- **深树**：能刻画复杂、非线性的关系（欠拟合风险低），但过拟合风险高、训练/预测计算成本更高、可解释性差。
- **浅树**：可解释性强、训练和预测快、不易过拟合，但可能欠拟合，无法捕捉复杂模式。

**解析**：

深度是控制**模型容量 (model capacity)** 的超参数：

- **过拟合/欠拟合**：
  - 深树 → 叶节点样本少，会拟合训练集中的噪声（高方差），训练误差低但测试误差可能高。
  - 浅树 → 区域划分太粗，连训练集都拟合不好（高偏差）。
- **计算成本**：
  - 训练：每层都要在各节点上枚举特征与阈值，层数越多总成本越高。
  - 预测：一次预测走一条根到叶的路径，代价与深度成正比，深度 $d$ 的树需要 $O(d)$ 次比较。
- **可解释性**：浅树的每条路径就是一条短小的 if-then 规则，人能直接读懂；深树规则又长又多，难以解释。

实践中用验证集选择合适的深度。

## 13. 决策树是否需要特征归一化、能否用于高维数据

**题目**：Recall that the KNN model suffers from the curse of dimensionality and it is a good idea to normalize the features before applying KNN. What about decision tree? Can we apply decision tree to high dimensional data?

**答案**：不需要归一化，因为分裂只依赖特征取值的相对顺序，与尺度无关。可以用于高维数据，因为分裂过程本身在做隐式特征选择；但当维度远多于样本数时仍可能过拟合。

**解析**：

- **归一化**：每次分裂只比较**单个特征**与阈值（$x_j \le t$）。对 $x_j$ 做任意单调变换（如缩放、平移 $x_j \mapsto a x_j + b,\ a>0$），样本的排序不变，能得到的划分集合不变，所以学到的树等价。而 kNN 依赖跨特征的距离 $\lVert \mathbf{x}-\mathbf{x}'\rVert$，尺度大的特征会主导距离，因此必须归一化。
- **高维数据**：kNN 的**维数灾难 (curse of dimensionality)** 在于高维空间中所有点彼此都"很远"、距离失去区分度。决策树不用距离，每次只挑信息增益最大的一个特征，无关特征基本不会被选中——相当于**隐式特征选择 (implicit feature selection)**。但特征极多而样本少时，某些无关特征可能碰巧在训练集上有高增益，导致过拟合。

## 14. 决策树能否处理缺失值

**题目**：Can decision tree handle missing values in a data set? For example, consider the lemon versus orange data set. Suppose that for one fruit, we know its weight but not its height.

**答案**：可以。许多决策树算法（如 C4.5、CART）能原生处理缺失值：用**替代分裂 (surrogate splits)**（用与缺失特征高度相关的其他特征代替），或按训练数据中各分支的比例把该样本**按概率/权重分配**到各子节点。

**解析**：

以柠檬/橙子为例，某水果已知重量 (weight) 但缺少高度 (height)：

- **训练时**：计算以 height 分裂的信息增益时，只用 height 已知的样本；缺失样本可按比例（带权重）同时送进两个子节点。
- **预测时**：遇到测试 height 的节点，
  - 替代分裂 (CART)：改用与 height 最相关的特征（如 weight）的分裂近似代替；
  - 概率分配 (C4.5)：把样本按训练时各分支的样本比例送入所有分支，最后把各叶的预测按权重加总。

相比之下，kNN 计算距离需要完整的特征向量，缺失值处理更麻烦。
