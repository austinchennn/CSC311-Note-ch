# Week 4 Lecture Prepare Quiz: Logistic Regression, Stochastic Gradient Descent, Multi-Class Classification

原卷：[week4_quiz.pdf](week4_quiz.pdf)

## 1. sigmoid 公式（单选）

**题目**：Which of the following is the correct formula for the sigmoid activation function $\sigma(z)$?

- A. $\sigma(z) = \frac{e^z}{e^z + e^{-z}}$
- B. $\sigma(z) = \max(0, z)$
- C. $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
- D. $\sigma(z) = \frac{1}{1 + e^{-z}}$

**答案**：D. $\sigma(z) = \frac{1}{1 + e^{-z}}$

**解析**：

$\sigma(z)=\frac1{1+e^{-z}}\in(0,1)$ 。B 是 ReLU，C 是 $\tanh$ ，A 其实等于 $\sigma(2z)$ 。

## 2. 交叉熵损失（单选）

**题目**：The cross-entropy loss $\mathcal{L}_{CE}(y,t)$ for logistic regression is defined as which expression?

- A. $\mathcal{L}_{CE}(y,t) = t \cdot y + (1-t) \cdot (1-y)$
- B. $\mathcal{L}_{CE}(y,t) = (y - t)^2$
- C. $\mathcal{L}_{CE}(y,t) = |y - t|$
- D. $\mathcal{L}_{CE}(y,t) = -t\log(y) - (1-t)\log(1-y)$

**答案**：D. $\mathcal{L}_{CE}(y,t) = -t\log(y) - (1-t)\log(1-y)$

**解析**：

交叉熵 (cross-entropy) $\mathcal{L}_{CE}=-t\log y-(1-t)\log(1-y)$ 。 $t=1$ 时只剩 $-\log y$ ， $t=0$ 时只剩 $-\log(1-y)$ 。预测越自信却越错，惩罚越接近无穷大。

## 3. 逻辑回归输出的含义（单选）

**题目**：The output $y = \sigma(\mathbf{w}^\top \mathbf{x})$ of a logistic regression model represents which quantity?

- A. The Euclidean distance from input $\mathbf{x}$ to the decision boundary
- B. The predicted class label assigned to input $\mathbf{x}$
- C. The probability of the positive class given input $\mathbf{x}$
- D. The magnitude of the weight vector $\mathbf{w}$ projected onto $\mathbf{x}$

**答案**：C. The probability of the positive class given input $\mathbf{x}$

**解析**：

$y=p(t=1\mid\mathbf{x})$ 是正类的概率。把 $y$ 与阈值 0.5 比较之后，才得到类别标签。

## 4. 平方误差做二分类的问题（单选）

**题目**：Why is the squared error loss problematic when applied to binary classification?

- A. It requires the model to output continuous values between negative and positive infinity
- B. It produces output values that are outside the range $(0,1)$
- C. It assigns infinite loss to every misclassified example
- D. It penalizes confident correct predictions

**答案**：D. It penalizes confident correct predictions

**解析**：

线性模型加平方误差时，如果 $t=1$ 而预测 $y=5$ ，方向判断得很对，但 $(5-1)^2$ 很大，模型反而因为“太正确”被惩罚，决策边界会被这些点拉偏。参见错题集 logistic_regression 第 2 题。

## 5. 逻辑回归相对 kNN 的优势（单选）

**题目**：Which of the following is an advantage of logistic regression over $k$-nearest neighbors?

- A. Logistic regression can learn complex nonlinear decision boundaries
- B. Logistic regression stores the entire training dataset for making predictions
- C. Logistic regression automatically increases model complexity with more training data
- D. Logistic regression provides fast predictions by using a fixed set of learned weights

**答案**：D. Logistic regression provides fast predictions by using a fixed set of learned weights

**解析**：

逻辑回归是参数模型 (parametric)，预测时只需算一次 $\sigma(\mathbf{w}^\top\mathbf{x})$ ，复杂度 $O(D)$ 。kNN 预测时要和全部 $N$ 个训练样本算距离。A、B、C 描述的都是 kNN 的特点。

## 6. 全批量梯度下降用多少样本（单选）

**题目**：In full batch gradient descent, the gradient of the cost function is computed using how many training examples per update?

- A. A small random subset of the training examples
- B. All $N$ training examples in the dataset
- C. Only the training examples that are currently misclassified
- D. A single randomly selected training example

**答案**：B. All $N$ training examples in the dataset

**解析**：

全批量 (full batch) 每次更新都用全部 $N$ 个样本算梯度。A 是 mini-batch SGD，D 是单样本 SGD。

## 7. mini-batch 的定义（单选）

**题目**：In stochastic gradient descent, a mini-batch is best described as which of the following?

- A. The set of all misclassified examples identified after each epoch
- B. The entire training dataset used in each gradient computation
- C. A single example selected in fixed sequential order from the dataset
- D. A small subset of $k$ examples used to estimate the gradient

**答案**：D. A small subset of $k$ examples used to estimate the gradient

**解析**：

mini-batch 是随机抽出的 $k$ 个样本，用它们的平均梯度估计全量梯度。这个估计是无偏的，而且计算成本低。

## 8. ravine（峡谷）的成因（单选）

**题目**：A ravine in the optimization landscape arises when which condition holds?

- A. The loss function is equally steep in every weight direction
- B. Training progresses smoothly as there is a clear direction to the minimum
- C. Different weight dimensions have very different gradient magnitudes
- D. The training data contains an unequal number of positive and negative examples

**答案**：C. Different weight dimensions have very different gradient magnitudes

**解析**：

峡谷 (ravine) 指某些方向很陡、另一些方向很平。这时梯度下降会在陡的方向来回震荡，在平的方向前进很慢。常见原因是特征尺度不一致，解决办法是特征标准化。

## 9. 学习率过大（单选）

**题目**：If the learning rate is set too large during gradient descent optimization, what is the likely outcome?

- A. The optimization diverges and the loss function increases without bound
- B. The optimization finds a better minimum with higher generalization accuracy
- C. The optimization converges more slowly toward the minimum of the loss
- D. The optimization path becomes smoother and more stable over time

**答案**：A. The optimization diverges and the loss function increases without bound

**解析**：

学习率过大时每一步都越过最低点，而且越走越远，损失发散。收敛变慢是学习率太小时的表现，对应 C。

## 10. SGD 噪声的好处（单选）

**题目**：How can the noisy gradient updates in SGD be beneficial during optimization?

- A. They can help the optimization escape flat regions such as plateaus
- B. They guarantee that the algorithm always finds the global minimum
- C. They completely eliminate the need for feature standardization
- D. They ensure convergence in fewer iterations than full batch descent

**答案**：A. They can help the optimization escape flat regions such as plateaus

**解析**：

梯度估计里的随机噪声能帮助参数跳出平台区 (plateau) 和较差的局部极小值，但不能保证找到全局最优。按迭代次数算，SGD 通常需要更多步，只是每一步便宜得多。

## 11. 多分类目标的表示（单选）

**题目**：In multi-class classification with $K$ classes, the target is represented as which of the following?

- A. A one-hot vector in $\mathbb{R}^K$ with a single $1$ and the rest $0$s
- B. A binary vector in $\{0,1\}^K$ with multiple entries equal to $1$
- C. An integer value from $1$ to $K$ stored as a single scalar
- D. A probability distribution over the $K$ classes that sums to one

**答案**：A. A one-hot vector in $\mathbb{R}^K$ with a single $1$ and the rest $0$s

**解析**：

目标用 one-hot 编码 $\mathbf{t}\in\{0,1\}^K$ ，只有真实类别那一位是 1。多个位置为 1 的是多标签 (multi-label) 问题；D 描述的是模型输出 $\mathbf{y}$ ，不是目标。

## 12. softmax 的输出（单选）

**题目**：The softmax function maps a vector of logits $\mathbf{z} \in \mathbb{R}^K$ to which type of output?

- A. A binary vector where the entry for the largest logit becomes $1$
- B. A probability distribution
- C. A normalized vector with unit Euclidean magnitude
- D. A vector of integer rankings from the best class to the worst class

**答案**：B. A probability distribution

**解析**：

$y_k=\frac{e^{z_k}}{\sum_{k'}e^{z_{k'}}}$ ，每一项都为正，而且总和为 1，所以是一个概率分布。A 描述的是 argmax（硬 one-hot）。

## 13. softmax 回归的决策边界（单选）

**题目**：The decision boundary between classes $i$ and $j$ in softmax regression is defined by which condition?

- A. The cross-entropy loss for class $i$ equals the loss for class $j$
- B. The softmax probability for class $i$ equals exactly $0.5$
- C. The logits for both classes are equal
- D. The weight vectors for classes $i$ and $j$ are perpendicular to each other

**答案**：C. The logits for both classes are equal

**解析**：

softmax 是单调的，所以 $y_i=y_j\iff z_i=z_j\iff(\mathbf{w}_i-\mathbf{w}_j)^\top\mathbf{x}+(b_i-b_j)=0$ ，这是一个超平面。有 $K>2$ 个类时，边界上的概率不一定是 0.5，所以 B 错。

## 14. 成对决策边界的数量（单选）

**题目**：For a softmax regression model with $K$ classes, how many pairwise decision boundaries exist?

- A. $K^2$ boundaries
- B. $K$ boundaries
- C. $K(K-1)/2$ boundaries
- D. $K - 1$ boundaries

**答案**：C. $K(K-1)/2$ boundaries

**解析**：

每一对类别 $(i,j)$ 有一条边界，一共有 $\binom K2=K(K-1)/2$ 对。
