# Week 1 — Nearest Neighbours

错题记录：

## 1. 监督学习的定义（Week 1 Lecture Prepare Quiz，单选）

**题目**：Which of the following best describes supervised learning?

- A. Learning patterns from unlabeled inputs by grouping similar examples
- B. Learning to map inputs to outputs using labelled input-output pairs
- C. Memorizing the training set so that all future predictions are exact
- D. Choosing features without needing target outputs

**答案**：B. Learning to map inputs to outputs using labelled input-output pairs

**解析**：

监督学习 (supervised learning) 用带标签的输入-输出对 $(\mathbf{x}^{(i)}, t^{(i)})$ 学习映射 $f:\mathbf{x}\mapsto t$ 。A 是无监督学习（聚类），C 是死记硬背，不能泛化，D 不需要目标值，也不是监督学习。

## 2. 哪个是分类任务（Week 1 Lecture Prepare Quiz，单选）

**题目**：Which of the following tasks is a classification task?

- A. Predicting tomorrow's highest temperature in Celsius
- B. Predicting a person's height from shoe size
- C. Predicting whether a leaf is oak or maple
- D. Predicting the age of a person in years from a social media message

**答案**：C. Predicting whether a leaf is oak or maple

**解析**：

分类 (classification) 的输出是离散类别，回归 (regression) 的输出是连续实数。橡树/枫树是两个类别；温度、身高、年龄都是数值，属于回归。

## 3. 符号 t 与 y 的区别（Week 1 Lecture Prepare Quiz，单选）

**题目**：What is the difference between $t$ and $y$ in the standard notation?

- A. $t$ is the predicted output and $y$ is the target output
- B. $t$ is the target output and $y$ is the predicted output
- C. $t$ is a feature and $y$ is a label
- D. $t$ is used only for regression and $y$ only for classification

**答案**：B. $t$ is the target output and $y$ is the predicted output

**解析**：

课程约定： $t$ 是目标值 (target)，即真实标签； $y$ 是模型的预测 (prediction)。损失函数比较的就是 $y$ 和 $t$ ，例如 $\frac12(y-t)^2$ 。

## 4. 假设 f 的含义（Week 1 Lecture Prepare Quiz，单选）

**题目**：Which of the following best matches the meaning of a hypothesis f?

- A. A rule or mapping that takes an input x and produces a prediction
- B. A dataset of labelled examples used during training
- C. A method for evaluating generalization on unseen data
- D. A collection of models that all share the same hyperparameters

**答案**：A. A rule or mapping that takes an input x and produces a prediction

**解析**：

假设 (hypothesis) $f$ 是一个具体的映射 $y=f(\mathbf{x})$ ，把输入变成预测。学习算法做的事，就是在假设空间里挑出一个好的 $f$ 。

## 5. 回归还是分类由什么决定（Week 1 Lecture Prepare Quiz，单选）

**题目**：Which quantity determines whether a supervised learning problem is regression or classification?

- A. The type of input data
- B. The number of features
- C. The kind of output being predicted
- D. The size of the training set

**答案**：C. The kind of output being predicted

**解析**：

问题类型只看输出：输出连续是回归，输出离散类别是分类。输入的类型、特征数、数据量都不决定问题类型。

## 6. 用哪个数据集选模型（Week 1 Lecture Prepare Quiz，单选）

**题目**：Which dataset should be used to choose between different models?

- A. Training set
- B. Validation set
- C. Test set
- D. Entire labeled dataset combined

**答案**：B. Validation set

**解析**：

三个数据集分工不同：训练集 (training set) 拟合参数，验证集 (validation set) 选超参数和模型，测试集 (test set) 只在最后用一次，估计泛化误差。用测试集选模型等于把它泄露进了训练过程，泛化估计会偏乐观。

## 7. 上标 (i) 的含义（Week 1 Lecture Prepare Quiz，单选）

**题目**：In the notation $x^{(i)}$, what does the superscript $(i)$ indicate?

- A. The $i$-th input vector in the dataset
- B. The $i$-th feature of a single example
- C. The $i$-th class label
- D. The exponent applied to the vector $\mathbf{x}$

**答案**：A. The $i$-th input vector in the dataset

**解析**：

上标 $(i)$ 表示第 $i$ 个训练样本 $\mathbf{x}^{(i)}$ ；下标 $x_j$ 才表示第 $j$ 个特征。两者组合起来， $x_j^{(i)}$ 就是第 $i$ 个样本的第 $j$ 个特征。

## 8. MNIST 的数据增强（Week 1 Lecture Prepare Quiz，单选）

**题目**：Which of the following is an example of data augmentation for MNIST?

- A. Changing every label to the most common class
- B. Replacing grayscale values with color channels
- C. Shifting or rotating the image
- D. Removing difficult examples from the training set

**答案**：C. Shifting or rotating the image

**解析**：

数据增强 (data augmentation) 对样本做保持标签不变的变换，比如平移、旋转，生成更多训练数据。改标签、删样本都不是增强；灰度图变彩色也不保持语义，同样不算。

## 9. kNN 的基本思想（Week 1 Lecture Prepare Quiz，单选）

**题目**：Which of the following is the correct idea behind k-nearest neighbours?

- A. Predict by calculating the distance to the centroid of each class
- B. Predict using the average feature vector of the training set
- C. Predict using the majority class among the k closest training examples
- D. Partition the training data into k clusters and assign the query point to the nearest cluster center

**答案**：C. Predict using the majority class among the k closest training examples

**解析**：

kNN 先找出离查询点最近的 $k$ 个训练样本，再对它们的标签做多数投票 (majority vote)。A 是最近质心分类器，D 是 k-means 聚类，都不是 kNN。

## 10. 方向比大小重要时用什么度量（Week 1 Lecture Prepare Quiz，单选）

**题目**：Which of the following is said to be especially useful when the direction of vectors matters more than their magnitude, such as in text classification?

- A. Euclidean distance
- B. Manhattan distance
- C. Chebyshev distance
- D. Cosine similarity

**答案**：D. Cosine similarity

**解析**：

余弦相似度 (cosine similarity) $\frac{\mathbf{a}^\top\mathbf{b}}{\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert}$ 只看两个向量的夹角，与长度无关。文本的词频向量会随文档长度变大，但主题由方向决定，所以适合用余弦相似度。

## 11. k 是什么（Week 1 Lecture Prepare Quiz，单选）

**题目**：The number k in k nearest neighbours is an example of what?

- A. Feature
- B. Label
- C. Hyperparameter
- D. Decision boundary

**答案**：C. Hyperparameter

**解析**：

$k$ 不是从训练数据里学出来的，而是训练前人为设定、在验证集上调的值，所以是超参数 (hyperparameter)。

## 12. 3-NN 与 1-NN 的决策边界（Week 1 Lecture Prepare Quiz，单选）

**题目**：Compared with 1-NN, a 3-NN typically has a decision boundary that is:

- A. Smoother and less sensitive to individual training points
- B. Identical whenever Euclidean distance is used
- C. More jagged because three points create more boundaries
- D. Guaranteed to achieve higher training accuracy

**答案**：A. Smoother and less sensitive to individual training points

**解析**：

$k$ 越大，单个噪声点对投票的影响越小，决策边界越平滑，越不容易过拟合。1-NN 的边界最锯齿化，训练精度通常最高（100%），所以 D 错。

## 13. 1-NN 训练精度 100%（Week 1 Lecture Prepare Quiz，判断）

**题目**：Unless there are identical points with different labels, 1-NN achieves 100% training accuracy.

- A. True
- B. False

**答案**：A. True

**解析**：

对训练点做预测时，离它最近的就是它自己（距离为 0），所以预测出的就是它自己的标签。唯一的例外是同一个位置有标签不同的重复点。

## 14. k = N 的极端情况（Week 1 Lecture Prepare Quiz，单选）

**题目**：What happens in the extreme case k=N, where N is the size of the training set?

- A. The model predicts the label of the nearest point only
- B. The model always predicts the most common class in the training set
- C. The model becomes equivalent to 1-NN
- D. The model achieves zero training error by construction

**答案**：B. The model always predicts the most common class in the training set

**解析**：

$k=N$ 时，每次投票都用上全部训练点，结果与查询点在哪里无关，模型总是预测训练集里最常见的类别，严重欠拟合 (underfitting)。

## 15. 实践中如何选 k（Week 1 Lecture Prepare Quiz，单选）

**题目**：In practice, how do we choose the value of k for a k nearest neighbour model for a given dataset?

- A. By always setting k=1
- B. By selecting the largest possible k
- C. By evaluating different values on a validation set
- D. By computing it directly from the test set

**答案**：C. By evaluating different values on a validation set

**解析**：

超参数在验证集上调：多试几个 $k$ ，选验证误差最小的那个。不能用测试集来选（理由见“用哪个数据集选模型”那题）。

## 16. 特征尺度不同对 kNN 的影响（Week 1 Lecture Prepare Quiz，单选）

**题目**：If feature A is measured on a much larger scale than feature B, then in k-NN:

- A. Differences in feature B have a larger effect on neighbor selection
- B. The algorithm automatically rescales all features before computing distances
- C. Both features always contribute equally to the distance calculation
- D. Differences in feature A can have a larger effect on distance calculations

**答案**：D. Differences in feature A can have a larger effect on distance calculations

**解析**：

欧氏距离 $\sqrt{\sum_j (x_j-x'_j)^2}$ 里，尺度大的特征差值大，会主导距离。kNN 不会自动缩放，所以通常先做标准化 (standardization)。

## 17. 训练集精度 100% 意味着什么（Week 1 Lecture Prepare Quiz，单选）

**题目**：A classifier has perfect accuracy on its training set. What is the most likely conclusion?

- A. It will necessarily achieve perfect test accuracy
- B. It may have overfit the training set
- C. It must be less complex than any classifier with lower training accuracy
- D. It should be trained on less data

**答案**：B. It may have overfit the training set

**解析**：

训练误差为 0 并不保证泛化好，反而可能是过拟合 (overfitting)：模型把噪声也记住了。泛化好不好要看验证集或测试集上的误差。

## 18. 高维空间中的欧氏距离（Week 1 Lecture Prepare Quiz，单选）

**题目**：In high dimensions, what happens to the Euclidean distances between randomly sampled points?

- A. Most points are clustered near the origin
- B. Most points are either very close or very far apart
- C. Most points are far apart from each other
- D. Distances decrease as dimensionality increases

**答案**：C. Most points are far apart from each other

**解析**：

维数灾难 (curse of dimensionality)：维度越高，空间体积指数增长，随机点之间大多相距很远，而且距离差不多（最近邻与最远邻的距离趋于相近）。

## 19. 维数灾难对 kNN 的影响（多选）（Week 1 Lecture Prepare Quiz，多选）

**题目**：Which of the following statements describe the curse of dimensionality in the context of k-NN? Select all that apply.

- A. k-NN may require much more data to make reliable predictions in high dimensions.
- B. Increasing the number of dimensions guarantees higher prediction accuracy
- C. In high dimensions, the nearest and farthest neighbors can have similar distances
- D. The curse of dimensionality disappears if k is chosen large enough.

**答案**：A. k-NN may require much more data to make reliable predictions in high dimensions.、C. In high dimensions, the nearest and farthest neighbors can have similar distances

**解析**：

高维时，要覆盖整个空间需要的样本量随维度指数增长，而且最近邻和最远邻的距离接近，“近邻”失去意义。增加维度不保证提升精度；调大 $k$ 也解决不了这个问题。
