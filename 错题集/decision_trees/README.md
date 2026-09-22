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
