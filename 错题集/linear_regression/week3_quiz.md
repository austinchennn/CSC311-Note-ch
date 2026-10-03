# Week 3 Lecture Prepare Quiz: Linear Regression and Gradient Descent

原卷：[week3_quiz.pdf](week3_quiz.pdf)

## 1. 模型参数（单选）

**题目**：A model parameter is best described as:

- A. A value the learning algorithm automatically learns from the training data
- B. A setting chosen before training and typically tuned using the validation data
- C. A value that determines the number of features in the input vector
- D. A value fixed by the test set to measure generalization performance

**答案**：A. A value the learning algorithm automatically learns from the training data

**解析**：

参数 (parameter)，比如 $\mathbf{w}$ 、 $b$ ，是从训练数据中学出来的。B 描述的是超参数 (hyperparameter)。

## 2. 损失函数衡量什么（单选）

**题目**：The loss function measures:

- A. The discrepancy between the prediction and the target value for one example
- B. The average discrepancy between predictions and targets across the entire training set
- C. The discrepancy between the model's and a perfect model's prediction
- D. The average discrepancy between the prediction and the target value for one batch

**答案**：A. The discrepancy between the prediction and the target value for one example

**解析**：

损失 (loss) $\mathcal{L}(y,t)$ 针对单个样本；对整个训练集取平均得到的是代价函数 (cost) $\mathcal{E}=\frac1N\sum_i\mathcal{L}(y^{(i)},t^{(i)})$ ，也就是 B 描述的东西。

## 3. 平方误差损失的形式（单选）

**题目**：The squared error loss used for regression is:

- A. $\frac{1}{2}\lvert y-t\rvert$
- B. $(y-t)^2+\frac{1}{2}$
- C. $\frac{1}{N}(y-t)^2$
- D. $\frac{1}{2}(y-t)^2$

**答案**：D. $\frac{1}{2}(y-t)^2$

**解析**：

$\mathcal{L}_{SE}=\frac12(y-t)^2$ ，系数 $\frac12$ 是为了求导后和平方的 2 抵消。 $\frac1N$ 出现在对所有样本求平均的代价函数里，不在单个样本的损失里。

## 4. 设计矩阵的定义（单选）

**题目**：The data matrix (design matrix) $\mathbf{X}$ is defined so that:

- A. Each row is a weight vector and each column is a different training label
- B. Each row contains the features for one example and each column contains one feature across all examples
- C. Each row contains the prediction error with each example and each column is a different model parameter
- D. Each row contains one feature across all examples and each column contains the features for one example

**答案**：B. Each row contains the features for one example and each column contains one feature across all examples

**解析**：

$\mathbf{X}\in\mathbb{R}^{N\times D}$ （加偏置列后是 $N\times(D+1)$ ）： $N$ 是样本数， $D$ 是特征数。每一行是一个样本，每一列是一个特征。

## 5. 线性模型的标量形式（单选）

**题目**：For a linear model with $D$ features, the scalar-form prediction is:

- A. $y=\sum_{j=1}^D w_j x_j + b$
- B. $y=\sum_{j=1}^D (w_j x_j)^2 + b$
- C. $y=\sum_{j=1}^D \frac{x_j}{w_j} + b$
- D. $y=\sum_{j=1}^D w_j + x_j + b$

**答案**：A. $y=\sum_{j=1}^D w_j x_j + b$

**解析**：

线性模型是每个特征乘以对应权重，求和后再加偏置 $b$ 。

## 6. 向量化预测公式（单选）

**题目**：With the design matrix $\mathbf{X}\in\mathbb{R}^{N\times(D+1)}$ and weights $\mathbf{w}\in\mathbb{R}^{(D+1)\times 1}$, the vectorized prediction equation is:

- A. $\mathbf{y}=\mathbf{w}\mathbf{X}$
- B. $\mathbf{y}=\mathbf{X}^\top \mathbf{w}$
- C. $\mathbf{y}=\mathbf{w}^\top \mathbf{X}$
- D. $\mathbf{y}=\mathbf{X}\mathbf{w}$

**答案**：D. $\mathbf{y}=\mathbf{X}\mathbf{w}$

**解析**：

检查维度： $(N\times(D+1))\cdot((D+1)\times1)=N\times1$ ，正好得到每个样本一个预测。其他三个选项的维度对不上。

## 7. 向量化的 MSE 代价（单选）

**题目**：The vectorized MSE cost function can be written as:

- A. $\mathcal{E}(\mathbf{w})=\frac{1}{2N}(\mathbf{X}\mathbf{w}-\mathbf{t})^\top(\mathbf{X}\mathbf{w}-\mathbf{t})$
- B. $\mathcal{E}(\mathbf{w})=\frac{1}{2N}(\mathbf{X}^\top\mathbf{w}-\mathbf{t})^\top(\mathbf{X}^\top\mathbf{w}-\mathbf{t})$
- C. $\mathcal{E}(\mathbf{w})=\frac{1}{2N}(\mathbf{X}\mathbf{t}-\mathbf{w})^\top(\mathbf{X}\mathbf{t}-\mathbf{w})$
- D. $\mathcal{E}(\mathbf{w})=\frac{1}{2N}(\mathbf{w}^\top\mathbf{X}-\mathbf{t})^\top(\mathbf{w}^\top\mathbf{X}-\mathbf{t})$

**答案**：A. $\mathcal{E}(\mathbf{w})=\frac{1}{2N}(\mathbf{X}\mathbf{w}-\mathbf{t})^\top(\mathbf{X}\mathbf{w}-\mathbf{t})$

**解析**：

残差向量是 $\mathbf{X}\mathbf{w}-\mathbf{t}\in\mathbb{R}^N$ ，它和自己的内积就是平方和，再乘以 $\frac1{2N}$ 得到 $\frac1{2N}\sum_i(y^{(i)}-t^{(i)})^2$ 。

## 8. MSE 的向量化梯度（单选）

**题目**：The vectorized gradient of the MSE cost given in the note is:

- A. $\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\frac{1}{N}\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{t})$
- B. $\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\frac{1}{N}\mathbf{X}(\mathbf{X}\mathbf{w}-\mathbf{t})$
- C. $\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\frac{1}{N}\mathbf{X}(\mathbf{X}^\top\mathbf{w}-\mathbf{t})$
- D. $\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\frac{1}{N}\mathbf{X}^\top(\mathbf{X}^\top\mathbf{w}-\mathbf{t})$

**答案**：A. $\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\frac{1}{N}\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{t})$

**解析**：

梯度的维度要和 $\mathbf{w}$ 一样，是 $(D+1)\times1$ 。 $\mathbf{X}^\top$ 的维度是 $(D+1)\times N$ ，乘以 $N\times1$ 的残差向量，维度正好对上。

## 9. 闭式解（单选）

**题目**：The closed-form (direct) solution for the weights is:

- A. $\mathbf{w}=(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{X}\mathbf{t}$
- B. $\mathbf{w}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{t}$
- C. $\mathbf{w}=\mathbf{X}^{-1}\mathbf{t}$
- D. $\mathbf{w}=\mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{t}$

**答案**：B. $\mathbf{w}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{t}$

**解析**：

令梯度为 0： $\mathbf{X}^\top\mathbf{X}\mathbf{w}=\mathbf{X}^\top\mathbf{t}$ （正规方程，normal equations），解出 $\mathbf{w}=(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{t}$ 。 $\mathbf{X}$ 一般不是方阵，所以不能直接求逆，C 错。

## 10. 梯度下降是什么（单选）

**题目**：Gradient descent is best described as an algorithm that does which of the following?

- A. Solves for the zeros of the gradient algebraically in one step
- B. Iteratively updates parameters in a direction that reduces the function being minimized
- C. Randomly samples parameter values until it finds the best one
- D. Updates the parameters stepwise in the direction of the global minimum of the function being minimized

**答案**：B. Iteratively updates parameters in a direction that reduces the function being minimized

**解析**：

梯度下降的更新是 $\mathbf{w}\leftarrow\mathbf{w}-\alpha\nabla J$ ，沿局部下降方向迭代。负梯度方向不一定指向全局最小值，所以 D 错；A 描述的是闭式解。

## 11. 梯度指向哪里（单选）

**题目**：For a differentiable function $J(\mathbf{w})$, the gradient $\nabla_{\mathbf{w}}J(\mathbf{w})$ points in which direction?

- A. The direction of steepest descent
- B. A direction that always points toward the global minimum
- C. The direction of steepest ascent
- D. A direction parallel to the contour lines

**答案**：C. The direction of steepest ascent

**解析**：

梯度指向函数上升最快的方向 (steepest ascent)，所以梯度下降要沿负梯度走。梯度与等高线垂直，不是平行，所以 D 错。

## 12. 学习率过大的后果（单选）

**题目**：If the learning rate is too large, what is a likely consequence?

- A. The cost must decrease at every iteration
- B. The algorithm becomes the closed-form solution
- C. Updates may overshoot the minimum and move farther away
- D. The gradient becomes perpendicular to the contour lines for the first time

**答案**：C. Updates may overshoot the minimum and move farther away

**解析**：

步长太大会越过最低点，在谷底两侧来回震荡，甚至发散。学习率太小则收敛很慢。

## 13. 特征映射的定义（单选）

**题目**：What is the mathematical definition of a feature mapping in the context of linear regression?

- A. A function that transforms each input vector into a new representation.
- B. A procedure that maps input features directly to their corresponding training targets.
- C. A strategy for reducing the total number of data points to limit computation time.
- D. An algorithm that selects the most important original features from a dataset.

**答案**：A. A function that transforms each input vector into a new representation.

**解析**：

特征映射 (feature mapping) $\boldsymbol{\psi}(\mathbf{x})$ ，例如多项式特征 $[1,x,x^2,\dots]$ ，把输入变换成新的表示。在新特征上仍然做线性回归，就能拟合非线性关系。

## 14. 正则化系数 λ 太小（单选）

**题目**：If the $\lambda$ in $\mathcal{E}(\mathbf{w}) + \lambda \mathcal{R}(\mathbf{w})$ is too small, then

- A. The regularizer dominates and the model may underfit
- B. The MSE term dominates and overfitting may remain
- C. The bias term is forced to zero
- D. The polynomial degree decreases automatically

**答案**：B. The MSE term dominates and overfitting may remain

**解析**：

$\lambda\to0$ 相当于没有正则化，MSE 项占主导，模型仍可能过拟合。 $\lambda$ 太大时正则项占主导，权重被压得太小，会欠拟合，这就是 A 描述的情况。

## 15. 正则化的含义（单选）

**题目**：Regularization is a modification of training that

- A. Chooses models using criteria beyond training error
- B. Forces all weights to have similar scale during training
- C. Guarantees zero training error on the dataset
- D. Removes the need to tune hyperparameters

**答案**：A. Chooses models using criteria beyond training error

**解析**：

正则化 (regularization) 在训练误差之外，再加入对模型复杂度的偏好，例如惩罚 $\lVert\mathbf{w}\rVert^2$ 。它会引入新的超参数 $\lambda$ ，所以 D 错。
