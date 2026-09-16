> 来源: https://www.teach.cs.toronto.edu/~csc311notes/intro.html

# Introduction to Machine Learning — 笔记

## 一、核心定义

**人工智能 (AI)**：研究能感知环境并采取行动的智能体（agent）的学科，目标是设计“行为理性”的智能体——即做出“最优”的动作或预测（“最优”因场景而异）。

**机器学习 (ML)**：AI 的子集，通过经验自动提升系统在某任务上的表现；本质是从数据中学习规律，而非手工为每个任务编程。

**深度学习 (Deep Learning)**：使用多层神经网络建模复杂数据模式的一类机器学习技术。

**大语言模型 (LLM)**：在海量文本上训练的深度学习模型，用于理解和生成类人语言；是深度学习在 NLP 上的一种具体应用。

## 二、五大核心思想

### 1. 学习即优化 (Learning is Optimization)
- 把“学习”转化为优化问题：定义**目标函数 / 损失函数 (objective function / loss function)** 衡量模型好坏，在给定的**模型空间 (model space)** 中寻找使目标函数最优的模型。
- 模型 = 数学表示 + 目标函数 + 优化算法 (optimization algorithm)，三者可自由组合 → ML 模型具有**模块化 (modularity)** 特性，也是理解新论文/新方法的统一视角。

### 2. 误差来源的权衡 (Sources of Error)
误差可分解为三类来源：
1. **欠拟合 (underfitting)**：模型过于简单，无法捕捉数据规律；
2. **过拟合 (overfitting)**：模型过于复杂，捕捉到数据中不能泛化的噪声模式；
3. **不可约误差 (irreducible error)**：问题本身固有的随机性/变异，模型无法消除。

- (1)(2) 之间的权衡 = **偏差-方差权衡 (bias-variance tradeoff)**（第6章形式化）。
- (3) 与特征/数据表示选择有关：特征太少或太多都会使模型变差 → 好的输入表示至关重要。
- 例：用鞋码预测身高——恒定预测平均身高（欠拟合）、记住个别噪声样本（过拟合）、同鞋码不同身高的固有变异（不可约误差）。

### 3. 模型评估是经验性的 (Model Evaluation is Empirical)
- **没有免费午餐定理 (No Free Lunch theorem)**：不存在普遍最优的算法，“最优”依赖具体场景，只能靠实验比较。
- 现实数据通常非独立同分布、非正态 → 渐近/理论保证不够，需在真实数据上做经验评估。
- 评估标准不止“准确率”：还包括效率、可解释性、鲁棒性、公平性等。
- 例：整体准确率相同，但对不同群体（性别、肤色、历史弱势群体）可能存在系统性偏差（癌症风险预测、人脸识别、简历筛选）。

### 4. ML 描述几何过程 (Geometric Processes)
- 数据点 = 高维空间中的向量：D 个特征 → $\mathbb{R}^D$ 中的点（例：叶片宽高 → $\mathbb{R}^2$ ）。
- 可用**距离 (distance)、相似度 (similarity)、变换 (transformation)** 等几何概念分析数据；坐标轴旋转不改变点间距离，说明表示与具体度量方式无关 → 支撑**分布式表示 (distributed representations)**、**词嵌入 (embeddings)** 等概念。
- 模型 = 对数据集的几何变换；损失函数与优化过程 = 在模型空间中寻找最优点的几何过程 (geometric process)。
- 结论：数据、模型、损失均可视为几何对象/过程（geometric objects/processes）—— 数据空间几何决定能学到什么，模型空间几何决定可选模型范围，损失曲面 (loss landscape) 几何决定能否找到好模型。
- 呼应**特征工程 (feature engineering) / 表示学习 (representation learning)**：特征选择决定数据的几何结构，从而决定哪些模式易学/难学。

### 5. ML 需要概率视角 (Probabilistic Lens)
- 真实数据本质上是从某概率分布采样的含噪声观测 → 不能期望模型做出完全确定性的预测。
- 理解不确定性可反过来用于建模：**数据增强 (data augmentation)**——主动给数据加噪声以提升模型鲁棒性。
- 许多学习算法/损失函数背后有概率解释（部分显式，部分隐式，如 k-means，第9章）。
- 优化过程常是**随机的 (stochastic)**：引入随机性有助于找到更好的解，提升对训练数据特性的鲁棒性 (robustness)。
- 几何视角 (geometric perspective) + 概率视角 (probabilistic perspective) = ML 两大数学基础：几何解释“结构与关系”，概率解释“不确定性与变异性”。
