> 来源: https://www.teach.cs.toronto.edu/~csc311notes/linear_classification/sgd.html

# Stochastic Gradient Descent — 笔记

## 一、核心定义

**全批量梯度下降 (full batch gradient descent)**：每次迭代都用**全部** $N$ 个训练样本计算精确梯度再更新参数；当 $N$ 很大（几百万级）时，每次迭代都要遍历全部数据，计算和内存开销都很高。

**小批量 (mini-batch / batch)**：每次迭代只随机抽取一个大小为 $k$ 的样本子集 $\mathcal{B}$ 来**估计**梯度；$k$ 称为**批大小 (batch size)**。

**随机梯度下降 (stochastic gradient descent, SGD)**：用小批量估计的梯度代替全批量精确梯度进行更新的梯度下降变体；"随机"指小批量的随机抽样过程给优化引入了噪声。

**迭代 (iteration)**：处理一个小批量、更新一次参数,称为一次迭代。

**轮 / epoch**：完整遍历一遍全部训练数据（处理完所有小批量）称为一个 epoch。

**平台 (plateau)**：损失曲面中梯度很小、但损失本身仍然很高的区域——在这种"平坦谷地"里参数更新幅度极小，优化进展缓慢（此前 sigmoid+平方误差在预测极端自信但错误时就是典型平台区）。

**峡谷 (ravine)**：损失曲面在某个方向上变化远比其垂直方向剧烈的区域，呈狭长山谷状；常由特征尺度差异悬殊或特征高度相关引起，会导致梯度下降在陡峭方向反复震荡、在平坦方向前进极慢。

## 二、公式

全批量梯度下降更新：

$$
\mathbf{w}\leftarrow\mathbf{w}-\alpha\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\mathbf{w}-\frac{\alpha}{N}\sum_{i=1}^N\nabla_{\mathbf{w}}\mathcal{L}(y^{(i)},t^{(i)})
$$

用小批量估计梯度：

$$
\nabla_{\mathbf{w}}\mathcal{E}(\mathbf{w})=\frac{1}{N}\sum_{i=1}^N\nabla_{\mathbf{w}}\mathcal{L}(y^{(i)},t^{(i)})\approx\frac{1}{k}\sum_{i\in\mathcal{B}}\nabla_{\mathbf{w}}\mathcal{L}(y^{(i)},t^{(i)})
$$

SGD 更新规则：

$$
\mathbf{w}\leftarrow\mathbf{w}-\frac{\alpha}{k}\sum_{i\in\mathcal{B}}\nabla_{\mathbf{w}}\mathcal{L}(y^{(i)},t^{(i)})
$$

标准化（与 kNN、线性回归中相同）：

$$
\tilde{x}_j=\frac{x_j-\mu_j}{\sigma_j}
$$

## 三、方法论：SGD 算法与超参数选择

**SGD 算法流程**：① 初始化参数；② 每个 epoch 内把训练集切成 $N/k$ 个互不重叠的小批量；③ 对每个小批量计算梯度估计并更新参数；④ 重复直到收敛或达到停止条件（如固定 epoch 数、代价不再下降）。

**批大小 (batch size) 的权衡**：批太大 → 和全批量梯度下降一样，单次迭代计算/存储代价高，但梯度估计更准，优化路径更平滑；批太小 → 梯度估计噪声大，路径更"抖动"，可能减慢收敛，但单次迭代更快、更省内存，且噪声有时能帮助跳出较差的局部极小值。实践中批大小常取 2 的幂次（如 32–256）。

**学习率 (learning rate) 的权衡**：太小 → 收敛慢；太大 → 可能发散或在最优点附近震荡。全批量梯度下降中学习率过大会让代价明显发散，但 SGD 由于梯度本身带噪声，过大的学习率不一定立刻发散，却会妨碍收敛到好的解。

**批大小与学习率的相互作用**：批越小、梯度噪声越大，通常需要**更小**的学习率来抵消噪声；批越大、梯度越准，往往能承受**更大**的学习率。二者需要联合调优，不能独立看待。
