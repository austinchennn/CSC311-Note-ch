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
