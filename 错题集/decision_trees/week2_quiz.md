# Week 2 Lecture Prepare Quiz: Decision Trees（Practice 版）

原卷：[week2_quiz.pdf](week2_quiz.pdf)

## 1. 什么是决策树（单选）

**题目**：What is a decision tree?

- A. A sequence of layered if-else statements used to make predictions on new data
- B. A method that always compares a test point to every training example in the dataset
- C. A model that can only be used for regression problems and not classification
- D. A random process for selecting class labels without using input features

**答案**：A. A sequence of layered if-else statements used to make predictions on new data

**解析**：

决策树 (decision tree) 是一层套一层的 if-else：每个内部节点测试一个条件，走到叶子节点得到预测。它既能做分类也能做回归；B 描述的是 kNN。

## 2. 非叶节点的作用（单选）

**题目**：What does a non-leaf node in a decision tree do?

- A. Aggregates the final predictions from its child nodes to produce an output
- B. Tests a condition on one feature to decide which child to follow
- C. Stores all training examples that reach the node for nearest-neighbor comparison
- D. Assigns a probability distribution over classes without making a split

**答案**：B. Tests a condition on one feature to decide which child to follow

**解析**：

内部节点 (internal node) 只对一个特征做测试，例如 $x_j\le\theta$ ，根据结果走向某个子节点。真正给出预测的是叶子节点。

## 3. 什么是决策边界（单选）

**题目**：What is a decision boundary?

- A. The set of training examples that are misclassified
- B. A set of lines that separates data perfectly based on their class
- C. A partition of the data space corresponding to predicted classes
- D. The maximum depth of a decision tree

**答案**：C. A partition of the data space corresponding to predicted classes

**解析**：

决策边界 (decision boundary) 把输入空间切分成若干区域，每个区域对应一个预测类别。它不一定能完美分开数据，所以 B 错。

## 4. 熵衡量什么（单选）

**题目**：What quantity is entropy used to measure?

- A. The number of nodes in a decision tree
- B. The uncertainty in a probability distribution
- C. The accuracy of k-nearest neighbours
- D. The amount of training data stored by a model

**答案**：B. The uncertainty in a probability distribution

**解析**：

熵 (entropy) $H(Y)=-\sum_y p(y)\log_2 p(y)$ 衡量一个分布的不确定性：分布越均匀熵越大，越确定熵越小。

## 5. 确定事件的熵（单选）

**题目**：If a random variable has one outcome with probability 1, what is its entropy in bits?

- A. $0$
- B. $1$
- C. $\infty$
- D. $-\infty$

**答案**：A. $0$

**解析**：

$H=-1\cdot\log_2 1=0$ （约定 $0\log 0=0$ ）。结果完全确定时没有不确定性，所以熵为 0。

## 6. 计算二元变量的熵（单选）

**题目**：For a binary random variable $Y$ with $P(Y=1)=0.75$ and $P(Y=0)=0.25$, which of the following correctly computes the entropy $H(Y)$ in bits?

- A. $H(Y) = -(0.75\log_2 0.25 + 0.25\log_2 0.75)$
- B. $H(Y) = 0.75\log_2 0.75 + 0.25\log_2 0.25$
- C. $H(Y) = 0.75\log_2 0.25 + 0.25\log_2 0.75$
- D. $H(Y) = -(0.75\log_2 0.75 + 0.25\log_2 0.25)$

**答案**：D. $H(Y) = -(0.75\log_2 0.75 + 0.25\log_2 0.25)$

**解析**：

按定义 $H(Y)=-\sum_y p(y)\log_2 p(y)$ ：每个概率乘以它自己的对数，最后加负号，结果约为 $0.811$ bits。B 少了负号，结果为负；A、C 把概率和对数配错了。

## 7. 条件熵衡量什么（单选）

**题目**：What does conditional entropy measure?

- A. The average amount of uncertainty remaining in $Y$ once the value of $X$ is known
- B. The prior uncertainty of $X$ before any information about $Y$ has been collected
- C. The total reduction in the entropy of $Y$ achieved by conditioning on $X$
- D. The combined entropy across variables $X$ and $Y$ simultaneously

**答案**：A. The average amount of uncertainty remaining in $Y$ once the value of $X$ is known

**解析**：

条件熵 (conditional entropy) $H(Y\mid X)=\sum_x p(x)H(Y\mid X=x)$ 是知道 $X$ 之后， $Y$ 平均还剩多少不确定性。C 描述的是信息增益，D 描述的是联合熵 $H(X,Y)$ 。

## 8. 计算 H(Y | X=x)（单选）

**题目**：Suppose $P(Y=1 \mid X=x)=0.9$ and $P(Y=0 \mid X=x)=0.1$. Which of the following correctly computes the conditional entropy $H(Y \mid X=x)$ in bits?

- A. $H(Y \mid X=x) = -(0.9\log_2 0.1 + 0.1\log_2 0.9)$
- B. $H(Y \mid X=x) = 0.9\log_2 0.1 + 0.1\log_2 0.9$
- C. $H(Y \mid X=x) = -(0.9\log_2 0.9 + 0.1\log_2 0.1)$
- D. $H(Y \mid X=x) = 0.9\log_2 0.9 + 0.1\log_2 0.1$

**答案**：C. $H(Y \mid X=x) = -(0.9\log_2 0.9 + 0.1\log_2 0.1)$

**解析**：

固定 $X=x$ 后，就是对条件分布求熵： $-\sum_y p(y\mid x)\log_2 p(y\mid x)\approx0.469$ bits。错误选项的配对方式和“计算二元变量的熵”那题一样。

## 9. 信息增益的定义（单选）

**题目**：What is the information gain about random variable $Y$ by observing random variable $X$ defined as?

- A. $H(X)+H(Y)$
- B. $H(Y \mid X)-H(Y)$
- C. $H(Y)-H(Y \mid X)$
- D. $H(X,Y)-H(X)$

**答案**：C. $H(Y)-H(Y \mid X)$

**解析**：

信息增益 (information gain) $IG(Y\mid X)=H(Y)-H(Y\mid X)$ 表示观测 $X$ 后 $Y$ 的不确定性减少了多少。B 刚好写反了，结果非正；D 其实等于 $H(Y\mid X)$ 。

## 10. 信息增益恒非负（判断）

**题目**：Information gain about random variable $Y$ by observing random variable $X$ is always nonnegative.

- A. True
- B. False

**答案**：A. True

**解析**：

$IG=H(Y)-H(Y\mid X)=I(X;Y)\ge0$ ：多知道一个信息，平均来说不会让不确定性增加 (information can't hurt)。等号在 $X$ 与 $Y$ 独立时成立。

## 11. 决策树边界总是轴对齐（判断）

**题目**：Decision tree decision boundaries are always axis-aligned because each split tests a threshold on a single feature.

- A. True
- B. False

**答案**：A. True

**解析**：

每次分裂只检查 $x_j\le\theta$ ，对应一个垂直于某条坐标轴的超平面，所以决策边界由轴对齐 (axis-aligned) 的矩形块拼成。

## 12. 只用训练误差做分裂准则的局限（单选）

**题目**：What is a limitation of using training error alone as a split criterion in a decision tree?

- A. It ignores the remaining uncertainty in the resulting child nodes
- B. It can only be used to evaluate binary classification problems
- C. It requires all input features to support a differentiable error metric
- D. It cannot be applied when the input features take continuous values

**答案**：A. It ignores the remaining uncertainty in the resulting child nodes

**解析**：

两种分裂可能误分类数相同，但其中一种让某个子节点变得更纯（不确定性更低），更有利于后续继续分裂。误分类率看不出这种差别，熵或信息增益可以。

## 13. 多数投票预测什么（单选）

**题目**：What does the majority vote choose to predict?

- A. The class with the smallest entropy in the full dataset
- B. The most common class among the training points in that region
- C. A randomly chosen class from the region
- D. The class associated with the nearest training point

**答案**：B. The most common class among the training points in that region

**解析**：

叶子节点（区域）内的预测，是落进这个区域的训练样本中出现次数最多的类别。

## 14. 贪心算法能否保证全局最优树（判断）

**题目**：A greedy decision tree algorithm guarantees the globally optimal tree.

- A. True
- B. False

**答案**：B. False

**解析**：

贪心算法每一步只选当前信息增益最大的分裂，不回头修改。找全局最优树是 NP-hard 问题，贪心只能得到一个够用的近似解。

## 15. 连续特征只需检查有限个阈值（单选）

**题目**：Why do decision tree algorithms only need to evaluate finitely many split thresholds for a continuous feature?

- A. Only thresholds between two observed values of the feature can change how the data points are partitioned
- B. Continuous features must first be converted into a small number of discrete categories before any splitting occurs
- C. Decision trees are limited to working with features that have a small and bounded range of values
- D. Every valid threshold must be chosen to exactly equal one of the feature values observed in the training data

**答案**：A. Only thresholds between two observed values of the feature can change how the data points are partitioned

**解析**：

$N$ 个样本在某个特征上最多有 $N$ 个不同取值。阈值只有跨过一个观测值时，划分结果才会改变，所以只需检查相邻观测值之间的位置（通常取中点），最多 $N-1$ 个候选。D 的说法太绝对。

## 16. 决策树的停止准则（多选）（多选）

**题目**：Which of the following can be used as a stopping criterion during decision tree learning? Select all that apply.

- A. Stop splitting when the tree reaches a predefined maximum depth
- B. Stop splitting when a node contains fewer than a minimum number of samples
- C. Stop splitting when the best split produces too little information gain
- D. Stop splitting when all remaining features have different numeric scales

**答案**：A. Stop splitting when the tree reaches a predefined maximum depth、B. Stop splitting when a node contains fewer than a minimum number of samples、C. Stop splitting when the best split produces too little information gain

**解析**：

常见的停止准则有三种：达到最大深度、节点样本数太少、最佳分裂的信息增益低于阈值。三者都能限制树的复杂度，防止过拟合。特征尺度与决策树无关（树对尺度不敏感），所以 D 不是停止准则。
