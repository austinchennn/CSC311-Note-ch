> 来源: https://www.teach.cs.toronto.edu/~csc311notes/neural_networks/backpropagation.html

# Backpropagation — 笔记

## 一、核心定义

#### 反向传播 (backpropagation)

一种高效计算神经网络中**所有**参数梯度的算法：把信息沿网络反向传播 (propagating information backward) 并复用中间计算结果 (reusing intermediate computations)，一次遍历就得到所有权重和偏置的偏导，计算成本大约等于一次前向传播 (roughly the same cost as one forward pass)。

#### 网络宏观架构符号 (architecture notation)

原文以 $M=3$ 的网络为例：两个隐藏层 + 一个输出层。

- $M$ ：网络的总层数（不计输入层），此例 $M=3$ 。
- 第 $0$ 层：输入层 (input layer)；第 $1$ 层：隐藏层 1 (hidden layer 1)；第 $2$ 层：隐藏层 2 (hidden layer 2)；第 $3$ 层：输出层 (output layer)。一般地，第 $1,\dots,M-1$ 层为隐藏层，第 $M$ 层为输出层。
- $D^{(m)}$ ：第 $m$ 层的单元（神经元）数量。 $D^{(0)}$ 是输入特征数， $D^{(1)}$ 是隐藏层 1 的单元数， $D^{(3)}$ 是输出个数。
- $D^{(3)}>1$ ：输出是一个向量，因此目标 $\mathbf{t}\in\mathbb{R}^{D^{(3)}}$ 也是向量。
- 图 1 示例：输入层 $3$ 个单元（ $D^{(0)}=3$ ）、隐藏层 1 有 $4$ 个、隐藏层 2 有 $3$ 个、输出层 $2$ 个（ $D^{(3)}=2$ ）；这些数字只作示意，推导对一般维度都成立。

#### 核心变量符号 (variable notation)

**读法规律**：上标括号 $(m)$ 表示所在**层数**，下标 $i,j$ 表示该层内的**第几个单元**。

- $a^{(m)}_i$ ：激活值 (activation)，第 $m$ 层第 $i$ 个单元的最终输出， $\mathbf{a}^{(m)}\in\mathbb{R}^{D^{(m)}}$ 。
  - 特例： $a^{(0)}_j$ 就是输入层的值，即原始输入特征 $x_j$ （ $\mathbf{a}^{(0)}=\mathbf{x}$ ）。
- $z^{(m)}_i$ ：预激活值 (pre-activation)，第 $m$ 层第 $i$ 个单元**经过激活函数之前**的值，即上一层输出的加权和加偏置， $\mathbf{z}^{(m)}\in\mathbb{R}^{D^{(m)}}$ 。
- $W^{(m)}_{i,j}$ ：权重 (weight)，连接第 $m-1$ 层第 $j$ 个单元到第 $m$ 层第 $i$ 个单元，整层 $\mathbf{W}^{(m)}\in\mathbb{R}^{D^{(m)}\times D^{(m-1)}}$ 。
  - **注意下标顺序**：方向是**从 $j$ 指向 $i$** （第一个下标是终点/当前层，第二个下标是起点/上一层）。它把上一层的 $a^{(m-1)}_j$ 缩放后送进当前层的 $z^{(m)}_i$ 。
- $b^{(m)}_i$ ：偏置 (bias)，第 $m$ 层第 $i$ 个单元的偏置项， $\mathbf{b}^{(m)}\in\mathbb{R}^{D^{(m)}}$ 。
- $\sigma^{(m)}$ ：第 $m$ 层的激活函数 (activation function)，逐元素作用于 $z^{(m)}_i$ 得到 $a^{(m)}_i=\sigma^{(m)}(z^{(m)}_i)$ 。
- $t_k$ ：目标 (target)，第 $k$ 个输出单元对应的真实值（标签）。
- $L$ ：逐单元损失 (per-unit loss)，如 $L(a^{(3)}_k,t_k)$ 衡量第 $k$ 个输出与目标 $t_k$ 的差异。
- $C$ ：总代价 (total cost)，单个样本上所有逐单元损失之和 $C=\sum_kL(a^{(M)}_k,t_k)$ 。
- 下标约定：推导中 $i$ 标记**当前层**的单元， $j$ 标记**下一层（更靠近输入的那层）**的单元， $k$ 常用来标记更上一层（更靠近输出）的单元。

#### 计算图 (computation graph)

表示前向传播中各标量如何相互依赖的**有向图**。每个节点是一个标量：权重、偏置、预激活 $z$ 、激活 $a$ 或总代价 $C$ ；每条从上游节点指向下游节点的有向边，表示下游量是以上游量为输入计算出来的。计算图只记录前向依赖，本身不涉及求导，是推导梯度的主要工具。

#### 扇出 (fan out)

计算图中若一个节点有**多于一条出边**——即其值被用作多于一个下游节点的输入——则称该节点扇出。当节点扇出且所有下游路径最终都到达 $C$ 时， $C$ 对该节点的导数是**所有出路径贡献之和**，这正是多变量链式法则刻画的内容。

#### 单变量链式法则 (single-variable chain rule)

设 $x,u,y$ 为标量， $u=g(x)$ 、 $y=f(u)$ ， $g,f$ 可微，则

$$
\frac{\partial y}{\partial x}=\frac{\partial y}{\partial u}\cdot\frac{\partial u}{\partial x}
$$

适用于路径**不分叉**的情形。

#### 多变量链式法则 (multivariate chain rule)

设 $u,v_1,\dots,v_m,y$ 为标量，每个 $v_k=g_k(u)$ ， $y=f(v_1,\dots,v_m)$ ，函数均可微，则

$$
\frac{\partial y}{\partial u}=\sum_{k=1}^{m}\frac{\partial y}{\partial v_k}\cdot\frac{\partial v_k}{\partial u}
$$

适用于节点**扇出**的情形。

#### 动态规划 (dynamic programming)

把复杂问题分解为**重叠子问题 (overlapping subproblems)**，按特定顺序每个子问题只求解一次，并把结果存储 (stored) 下来复用 (reused) 的技术。在反向传播中，每个节点的子问题就是" $C$ 对该节点量的偏导"。

#### 前向传播 (forward pass) / 反向传播 (backward pass)

- **前向传播**：从第 $1$ 层到第 $M$ 层依次计算并**存储**每层的 $\mathbf{z}^{(m)},\mathbf{a}^{(m)}$ 。
- **反向传播**：从代价 $C$ 出发，沿与前向相反的方向应用链式法则，逐层向输入方向计算并存储导数，得到所有参数的梯度。

## 二、公式

### 1. 线性模型的回顾（训练的出发点）

$N$ 个训练样本， $t^{(i)}$ 为第 $i$ 个样本的目标，输入 $\mathbf{x}^{(i)}$ 、权重向量 $\mathbf{w}$ 、偏置 $b$ ：

$$
y^{(i)}=\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}+b),\qquad
\mathcal{E}(\mathbf{w},b)=\frac{1}{N}\sum_{i=1}^{N}\mathcal{L}(y^{(i)},t^{(i)})
$$

$$
\mathbf{w}\leftarrow\mathbf{w}-\eta\,\nabla_{\mathbf{w}}\mathcal{E},\qquad b\leftarrow b-\eta\,\frac{\partial\mathcal{E}}{\partial b}
$$

其中 $\eta>0$ 为学习率 (learning rate)。神经网络沿用完全相同的优化方式，难点只在于**如何高效地**对每个参数求偏导。

### 2. 前向传播（标量形式）

对 $m=1,\dots,M$ ：

$$
z^{(m)}_i=\sum_{j=1}^{D^{(m-1)}}W^{(m)}_{i,j}\,a^{(m-1)}_j+b^{(m)}_i,\qquad a^{(m)}_i=\sigma^{(m)}\bigl(z^{(m)}_i\bigr),\qquad i=1,\dots,D^{(m)}
$$

代价按输出单元分解：

$$
C=\sum_{k=1}^{D^{(M)}}L\bigl(a^{(M)}_k,t_k\bigr)\quad\Longrightarrow\quad\frac{\partial C}{\partial a^{(M)}_k}=\frac{\partial L(a^{(M)}_k,t_k)}{\partial a^{(M)}_k}
$$

（每项只依赖第 $k$ 个输出激活和第 $k$ 个目标，其余项对 $a^{(M)}_k$ 是常数。）

### 3. 反向传播递归公式（标量形式）

- **输出层（基本情况）**：

$$
\frac{\partial C}{\partial z^{(M)}_i}=\frac{\partial C}{\partial a^{(M)}_i}\cdot\sigma^{(M)\prime}\bigl(z^{(M)}_i\bigr),\qquad i=1,\dots,D^{(M)}
$$

- **隐藏层（递归情况）**，对 $m=M-1,\dots,1$ ：

$$
\frac{\partial C}{\partial z^{(m)}_i}=\left(\sum_{k=1}^{D^{(m+1)}}\frac{\partial C}{\partial z^{(m+1)}_k}\cdot W^{(m+1)}_{k,i}\right)\cdot\sigma^{(m)\prime}\bigl(z^{(m)}_i\bigr),\qquad i=1,\dots,D^{(m)}
$$

  括号内的和就是 $\partial C/\partial a^{(m)}_i$ （ $a^{(m)}_i$ 扇出到上一层所有 $z^{(m+1)}_k$ ）。

- **权重与偏置梯度**：

$$
\frac{\partial C}{\partial W^{(m)}_{i,j}}=\frac{\partial C}{\partial z^{(m)}_i}\cdot a^{(m-1)}_j,\qquad
\frac{\partial C}{\partial b^{(m)}_i}=\frac{\partial C}{\partial z^{(m)}_i}
$$

  $i=1,\dots,D^{(m)}$ ， $j=1,\dots,D^{(m-1)}$ 。

> 注：原文第 6 节和 7.1 节把"权重与偏置梯度"一步的循环写成 $m=M-1,\dots,1$ ，但按第 5.1 节（输出层 $W^{(3)},b^{(3)}$ 的梯度同样用这两个公式）以及下面的梯度下降更新，这里实际应覆盖**所有层** $m=M,\dots,1$ 。这是笔者的理解，原文写法如上。

### 4. 梯度下降更新

$$
W^{(m)}_{i,j}\leftarrow W^{(m)}_{i,j}-\eta\cdot\frac{\partial C}{\partial W^{(m)}_{i,j}},\qquad
b^{(m)}_i\leftarrow b^{(m)}_i-\eta\cdot\frac{\partial C}{\partial b^{(m)}_i}
$$

### 5. 向量化结果 (vectorization)

**前向**（原文写 $m=1,\dots,M-1$ ，按 7.1 节标量版应为 $m=1,\dots,M$ ）：

$$
\mathbf{z}^{(m)}=\mathbf{W}^{(m)}\mathbf{a}^{(m-1)}+\mathbf{b}^{(m)},\qquad\mathbf{a}^{(m)}=\sigma^{(m)}(\mathbf{z}^{(m)})
$$

**反向**：

$$
\nabla_{\mathbf{z}^{(M)}}C=(\nabla_{\mathbf{a}^{(M)}}C)\odot\sigma^{(M)\prime}(\mathbf{z}^{(M)})
$$

$$
\nabla_{\mathbf{z}^{(m)}}C=(\mathbf{W}^{(m+1)})^\top(\nabla_{\mathbf{z}^{(m+1)}}C)\odot\sigma^{(m)\prime}(\mathbf{z}^{(m)}),\qquad m=M-1,\dots,1
$$

$$
\nabla_{\mathbf{W}^{(m)}}C=(\nabla_{\mathbf{z}^{(m)}}C)(\mathbf{a}^{(m-1)})^\top,\qquad\nabla_{\mathbf{b}^{(m)}}C=\nabla_{\mathbf{z}^{(m)}}C
$$

维度检查：

| 量 | 形状 | 用到的运算 |
|---|---|---|
| $\mathbf{W}^{(m)}\mathbf{a}^{(m-1)}$ | $(D^{(m)}\times D^{(m-1)})(D^{(m-1)})\to D^{(m)}$ | 矩阵-向量积 (matrix-vector product) |
| $\nabla_{\mathbf{a}^{(M)}}C\odot\sigma^{(M)\prime}(\mathbf{z}^{(M)})$ | $D^{(M)}\odot D^{(M)}\to D^{(M)}$ | 逐元素积 / Hadamard 积 (element-wise / Hadamard product) |
| $(\mathbf{W}^{(m+1)})^\top\nabla_{\mathbf{z}^{(m+1)}}C$ | $(D^{(m)}\times D^{(m+1)})(D^{(m+1)})\to D^{(m)}$ | 矩阵-向量积 |
| $(\nabla_{\mathbf{z}^{(m)}}C)(\mathbf{a}^{(m-1)})^\top$ | $(D^{(m)})(D^{(m-1)})^\top\to D^{(m)}\times D^{(m-1)}$ | 外积 (outer product) |

公式虽少，向量化却用到了全部三种运算：矩阵-向量积、逐元素积、外积。

## 三、推导过程

以 $M=3$ 的网络为例（两个隐藏层 + 向量输出，要求 $D^{(3)}>1$ ，这是能暴露反向传播需处理的所有链式结构的最小架构），以非向量化形式手算三个标量权重梯度，一个比一个深。

### 1. 输出层权重 $W^{(3)}_{1,2}$ （无扇出）

路径： $W^{(3)}_{1,2}\to z^{(3)}_1\to a^{(3)}_1\to C$ ，每个节点恰好一条出边，只需单变量链式法则：

$$
\frac{\partial C}{\partial W^{(3)}_{1,2}}
=\underbrace{\frac{\partial L(a^{(3)}_1,t_1)}{\partial a^{(3)}_1}}_{\partial C/\partial a^{(3)}_1}
\cdot\underbrace{\sigma^{(3)\prime}\bigl(z^{(3)}_1\bigr)}_{\partial a^{(3)}_1/\partial z^{(3)}_1}
\cdot\underbrace{a^{(2)}_2}_{\partial z^{(3)}_1/\partial W^{(3)}_{1,2}}
$$

三因子 = 损失对输出激活的导数 × 激活函数在预激活处的导数 × 喂入该权重的上游激活。对任意 $W^{(3)}_{i,j}$ 结构不变，只换下标。偏置把最后一个因子换成 $1$ （ $\partial z^{(3)}_1/\partial b^{(3)}_1=1$ ）：

$$
\frac{\partial C}{\partial b^{(3)}_1}=\frac{\partial L(a^{(3)}_1,t_1)}{\partial a^{(3)}_1}\cdot\sigma^{(3)\prime}\bigl(z^{(3)}_1\bigr)
$$

### 2. 隐藏层 2 权重 $W^{(2)}_{3,4}$ （一次扇出）

路径： $W^{(2)}_{3,4}\to z^{(2)}_3\to a^{(2)}_3\to\{z^{(3)}_1,\dots,z^{(3)}_{D^{(3)}}\}\to C$ ， $a^{(2)}_3$ 处第一次出现扇出。

- **步骤 1（不分叉部分，单变量链式法则）**：

$$
\frac{\partial C}{\partial W^{(2)}_{3,4}}=\frac{\partial C}{\partial a^{(2)}_3}\cdot\sigma^{(2)\prime}\bigl(z^{(2)}_3\bigr)\cdot a^{(1)}_4
$$

- **步骤 2（扇出部分，多变量链式法则）**：取 $u=a^{(2)}_3$ 、 $v_k=z^{(3)}_k$ 、 $y=C$ ，

$$
\frac{\partial C}{\partial a^{(2)}_3}=\sum_{k=1}^{D^{(3)}}\frac{\partial C}{\partial z^{(3)}_k}\cdot W^{(3)}_{k,3},\qquad
\frac{\partial C}{\partial z^{(3)}_k}=\frac{\partial L(a^{(3)}_k,t_k)}{\partial a^{(3)}_k}\cdot\sigma^{(3)\prime}\bigl(z^{(3)}_k\bigr)
$$

- **代回**：

$$
\frac{\partial C}{\partial W^{(2)}_{3,4}}=\left(\sum_{k=1}^{D^{(3)}}\frac{\partial L(a^{(3)}_k,t_k)}{\partial a^{(3)}_k}\cdot\sigma^{(3)\prime}\bigl(z^{(3)}_k\bigr)\cdot W^{(3)}_{k,3}\right)\cdot\sigma^{(2)\prime}\bigl(z^{(2)}_3\bigr)\cdot a^{(1)}_4
$$

相比例 1，多出的结构只有扇出：扇出前仍是单链，扇出处变成每个输出单元一项的求和；每项内的 $\partial C/\partial z^{(3)}_k$ 正是例 1 的输出层导数，只是现在每个 $k$ 都要用到。

### 3. 隐藏层 1 权重 $W^{(1)}_{2,3}$ （两层扇出，原文留作练习并给出结果）

路径： $W^{(1)}_{2,3}\to z^{(1)}_2\to a^{(1)}_2\to\{z^{(2)}_i\}_{i=1}^{D^{(2)}}\to\{z^{(3)}_k\}_{k=1}^{D^{(3)}}\to C$ 。

- **步骤 1（第一次扇出）**： $\displaystyle\frac{\partial C}{\partial a^{(1)}_2}=\sum_{i=1}^{D^{(2)}}\frac{\partial C}{\partial z^{(2)}_i}\cdot W^{(2)}_{i,2}$
- **步骤 2（第二次扇出）**： $\displaystyle\frac{\partial C}{\partial z^{(2)}_i}=\frac{\partial C}{\partial a^{(2)}_i}\cdot\sigma^{(2)\prime}\bigl(z^{(2)}_i\bigr)=\left(\sum_{k=1}^{D^{(3)}}\frac{\partial C}{\partial z^{(3)}_k}\cdot W^{(3)}_{k,i}\right)\sigma^{(2)\prime}\bigl(z^{(2)}_i\bigr)$
- **步骤 3（合并）**：

$$
\frac{\partial C}{\partial W^{(1)}_{2,3}}=\left[\sum_{i=1}^{D^{(2)}}\left(\sum_{k=1}^{D^{(3)}}\frac{\partial C}{\partial z^{(3)}_k}\cdot W^{(3)}_{k,i}\right)\sigma^{(2)\prime}\bigl(z^{(2)}_i\bigr)\cdot W^{(2)}_{i,2}\right]\cdot\sigma^{(1)\prime}\bigl(z^{(1)}_2\bigr)\cdot a^{(0)}_3
$$

**暴露的冗余**：内层对 $k$ 的和要对每个 $i$ 计算一次，而 $\partial C/\partial z^{(3)}_k$ 与 $i$ 无关，却在直接计算中被从头重算 $D^{(2)}$ 次；同样 $\partial C/\partial a^{(2)}_i=\sum_k\frac{\partial C}{\partial z^{(3)}_k}W^{(3)}_{k,i}$ 每次经过第 2 层激活求导都要重算。推广到所有参数，独立计算每个 $\partial C/\partial W^{(m)}_{i,j}$ 会无数次重复同样的反向链。

### 4. 用一次反向遍历算出全部梯度（ $M=3$ ）

按 输出层 → 隐藏层 2 → 隐藏层 1 的顺序：

- **输出层**：代价是逐单元损失之和， $a^{(3)}_i$ 只出现在第 $i$ 项；乘非线性导数得 $\partial C/\partial z^{(3)}_i$ 并**存储**，权重/偏置梯度随即可得：

$$
\frac{\partial C}{\partial z^{(3)}_i}=\frac{\partial C}{\partial a^{(3)}_i}\sigma^{(3)\prime}\bigl(z^{(3)}_i\bigr),\quad
\frac{\partial C}{\partial W^{(3)}_{i,j}}=\frac{\partial C}{\partial z^{(3)}_i}a^{(2)}_j,\quad
\frac{\partial C}{\partial b^{(3)}_i}=\frac{\partial C}{\partial z^{(3)}_i}
$$

- **隐藏层 2**： $a^{(2)}_i$ 扇出到每个 $z^{(3)}_k$ ，直接**查表**使用已存的 $\partial C/\partial z^{(3)}_k$ ：

$$
\frac{\partial C}{\partial z^{(2)}_i}=\left(\sum_{k=1}^{D^{(3)}}\frac{\partial C}{\partial z^{(3)}_k}W^{(3)}_{k,i}\right)\sigma^{(2)\prime}\bigl(z^{(2)}_i\bigr),\quad
\frac{\partial C}{\partial W^{(2)}_{i,j}}=\frac{\partial C}{\partial z^{(2)}_i}a^{(1)}_j,\quad
\frac{\partial C}{\partial b^{(2)}_i}=\frac{\partial C}{\partial z^{(2)}_i}
$$

- **隐藏层 1**：同一模式，使用已存的 $\partial C/\partial z^{(2)}_k$ 和输入 $a^{(0)}_j$ ：

$$
\frac{\partial C}{\partial z^{(1)}_i}=\left(\sum_{k=1}^{D^{(2)}}\frac{\partial C}{\partial z^{(2)}_k}W^{(2)}_{k,i}\right)\sigma^{(1)\prime}\bigl(z^{(1)}_i\bigr),\quad
\frac{\partial C}{\partial W^{(1)}_{i,j}}=\frac{\partial C}{\partial z^{(1)}_i}a^{(0)}_j,\quad
\frac{\partial C}{\partial b^{(1)}_i}=\frac{\partial C}{\partial z^{(1)}_i}
$$

三层结构完全相同：每层先算并存储 $\partial C/\partial z^{(m)}_i$ ；权重/偏置梯度直接用它；隐藏层的 $\partial C/\partial a^{(m)}_i$ 通过 $W^{(m+1)}$ 上的加权和汇集上一层的贡献。每项只算一次——推广到一般 $M$ 即得第二节第 3 条的递归公式。

### 5. 向量化的推导（原文以选择题形式给出）

- **前向** $z^{(m)}_i=\sum_jW^{(m)}_{i,j}a^{(m-1)}_j+b^{(m)}_i$ ：由 $\mathbf{W}^{(m)}\in\mathbb{R}^{D^{(m)}\times D^{(m-1)}}$ 、 $\mathbf{a}^{(m-1)}\in\mathbb{R}^{D^{(m-1)}}$ 、结果需 $\in\mathbb{R}^{D^{(m)}}$ ，只能是 $\mathbf{W}^{(m)}\mathbf{a}^{(m-1)}$ ，再加 $\mathbf{b}^{(m)}\in\mathbb{R}^{D^{(m)}}$ 。
- **输出层** $\partial C/\partial z^{(M)}_i=\partial C/\partial a^{(M)}_i\cdot\sigma^{(M)\prime}(z^{(M)}_i)$ ：两者和结果都 $\in\mathbb{R}^{D^{(M)}}$ ，每个 $i$ 独立贡献 → 逐元素积 $\odot$ 。
- **隐藏层括号内加权和** $\sum_k\frac{\partial C}{\partial z^{(m+1)}_k}W^{(m+1)}_{k,i}$ ：对 $W^{(m+1)}$ 的**第一个下标** $k$ 求和、保留第二个下标 $i$ ，即 $\mathbf{W}^{(m+1)}$ 的第 $i$ 列与梯度的内积 → $(\mathbf{W}^{(m+1)})^\top\nabla_{\mathbf{z}^{(m+1)}}C\in\mathbb{R}^{D^{(m)}}$ ；再与 $\sigma^{(m)\prime}(\mathbf{z}^{(m)})$ 做 $\odot$ 。
- **权重梯度** $\partial C/\partial W^{(m)}_{i,j}=\frac{\partial C}{\partial z^{(m)}_i}a^{(m-1)}_j$ ：结果需 $\in\mathbb{R}^{D^{(m)}\times D^{(m-1)}}$ ，用两个向量构造矩阵，自然是外积 $(\nabla_{\mathbf{z}^{(m)}}C)(\mathbf{a}^{(m-1)})^\top$ 。

## 四、方法论

1. **训练神经网络的真正难点**：
   - 不在于"优化什么、怎么设置梯度下降"——这和线性模型完全一样（初始化 → 求偏导 → 沿降低代价方向更新）。
   - 而在于**高效**求偏导：网络可有百万级参数，每个偏导独立计算意味着每步百万次计算，实践中不可行 (computationally infeasible)。
2. **推导单个梯度的步骤**：
   - 先写出前向传播，画计算图，追踪从参数到 $C$ 的所有路径。
   - 路径不分叉 → 单变量链式法则（连乘）。
   - 节点扇出 → 多变量链式法则（对所有出路径求和）。
3. **完整的反向传播算法（Complete Version）**：
   - **前向传播**：对 $m=1,\dots,M$ 计算并存储 $z^{(m)}_i$ 、 $a^{(m)}_i$ 。
   - **反向传播**：先算输出层 $\partial C/\partial z^{(M)}_i$ ，再对 $m=M-1,\dots,1$ 递归算 $\partial C/\partial z^{(m)}_i$ ，并由它得到各层 $\partial C/\partial W^{(m)}_{i,j}$ 、 $\partial C/\partial b^{(m)}_i$ 。
   - **梯度下降更新**：用学习率 $\eta>0$ 更新所有 $W^{(m)}$ 、 $b^{(m)}$ 。
4. **为什么前向传播必须先于反向传播**：检查反向公式所需的量——
   - $\partial C/\partial a^{(M)}_i$ ：固定损失函数后即可确定，能预先准备。
   - $\sigma^{(m)\prime}(z^{(m)}_i)$ ：固定每层激活函数后导数形式即确定，能预先准备。
   - $W^{(m)}$ ：本来就是要更新的量，始终保存着。
   - $a^{(m-1)}_j$ ：**无法预先准备**——它依赖具体输入，每个训练样本都不同；所以必须先跑前向传播，计算并存储各层中间激活，供反向传播使用。
5. **向量化**：实践中都要把反向传播向量化以提高效率；做法是先写出每个量的形状，再由"结果应有的形状"反推该用矩阵-向量积、逐元素积还是外积。

## 五、核心概念与思想

- **动态规划视角 (the dynamic-programming view of backpropagation)**：反向传播的关键洞见——不从头重算每个梯度，而是在每个**预激活节点**存储中间导数 $\partial C/\partial z^{(m)}_i$ ，并复用给所有喂入该节点的权重；每个子问题只解一次，一次反向传播（计算量大致等于一次前向传播）同时得到所有权重和偏置的梯度。
- **递归调用发生在预激活节点**：反向传播是一个递归过程 (recursive procedure)，其递归发生在每层的预激活 $\mathbf{z}^{(m)}$ 处——算出 $\nabla_{\mathbf{z}^{(m)}}C$ 后，本层参数梯度（乘 $\mathbf{a}^{(m-1)}$ 或直接取用）和下一层的 $\nabla_{\mathbf{z}^{(m-1)}}C$ （乘 $(\mathbf{W}^{(m)})^\top$ 再 $\odot\sigma'$ ）都由它得出。
- **权重梯度 = 下游误差信号 × 上游激活**： $\partial C/\partial W^{(m)}_{i,j}=\frac{\partial C}{\partial z^{(m)}_i}\cdot a^{(m-1)}_j$ ，偏置梯度就是 $\partial C/\partial z^{(m)}_i$ 本身。
- **Idea #1（学习即优化，learning is optimization）**：是反向传播存在的原因——训练神经网络就是在百万参数上最小化代价，反向传播提供梯度，使这一优化变得可行 (tractable)。
- **Idea #2（平衡误差来源的权衡，balancing tradeoffs in sources of error）**：支配算法到手后的每个实践选择——更多层/更多单元能更紧地拟合训练数据，但有过拟合风险；正则化 (regularization) 和谨慎的初始化 (careful initialization) 是应对这一权衡的工具。
- **Idea #3（模型评估是经验性的，model evaluation is empirical）**：解释了为什么层数、每层单元数、激活函数等架构决策没有通用答案——唯一可靠的指南是在具体问题和数据集上做经验评估。

## 六、例子（精简保留）

- **示意网络（图 1）**：输入层 3 单元 → 隐藏层 1 有 4 单元 → 隐藏层 2 有 3 单元 → 输出层 2 单元；单元数仅作示意，推导对一般 $D^{(0)},D^{(1)},D^{(2)},D^{(3)}$ （ $D^{(3)}>1$ ）成立。
- **例 1 输出层权重 $W^{(3)}_{1,2}$ （图 2）**：路径不分叉，梯度 = $\frac{\partial L}{\partial a^{(3)}_1}\cdot\sigma^{(3)\prime}(z^{(3)}_1)\cdot a^{(2)}_2$ 。
- **例 2 隐藏层 2 权重 $W^{(2)}_{3,4}$ （图 3）**： $a^{(2)}_3$ 扇出到所有 $z^{(3)}_k$ ，梯度含一个对 $k$ 的求和。
- **例 3 隐藏层 1 权重 $W^{(1)}_{2,3}$ （图 4）**：两层扇出，梯度含对 $i$ 、 $k$ 的嵌套求和， $\partial C/\partial z^{(3)}_k$ 被重算 $D^{(2)}$ 次——冗余的直接证据。
- **向量化计算图（图 5–8）**：节点为整层量 $\mathbf{a}^{(m)},\mathbf{W}^{(m)},\mathbf{b}^{(m)},\mathbf{z}^{(m)}$ ；反向传播从右端 $C$ 逆向遍历，依次高亮输出层、隐藏层 2、隐藏层 1 的节点。
- **向量化选择题**（4 道，答案）：
  - 前向 $\mathbf{z}^{(m)}$ → (C) $\mathbf{W}^{(m)}\mathbf{a}^{(m-1)}+\mathbf{b}^{(m)}$ 。
  - 输出层 $\nabla_{\mathbf{z}^{(M)}}C$ → (C) $(\nabla_{\mathbf{a}^{(M)}}C)\odot\sigma^{(M)\prime}(\mathbf{z}^{(M)})$ 。
  - 隐藏层括号内加权和 → (B) $(\mathbf{W}^{(m+1)})^\top(\nabla_{\mathbf{z}^{(m+1)}}C)$ 。
  - 权重梯度 $\nabla_{\mathbf{W}^{(m)}}C$ → (E) $(\nabla_{\mathbf{z}^{(m)}}C)(\mathbf{a}^{(m-1)})^\top$ 。
