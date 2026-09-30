# 闲言碎语

课堂外零散的问答、补充阅读记录。

## 集成学习（Ensemble Learning）

教授原话：

> Yes, this is an acceptable approach, and I would consider it a form of ensemble learning. What you described is called heterogeneous ensembles. You are combining predictions from multiple models, even though the models use different algorithms and different subsets of features. I suggest that you think carefully about how to partition the features. This is a design choice and should be made intentionally rather than arbitrarily.
>
> If you want to explore related ideas, you can look into voting ensembles and stacking. Bagging and boosting are also examples of ensemble learning, although they work somewhat differently from the approach you described. We will cover bagging briefly in week 7.

要点：
- 用**不同算法 + 不同特征子集**训练多个模型再合并预测 → **异质集成（heterogeneous ensemble）**。
- 特征如何划分是设计选择，要有理由，不能随便分。
- 相关方法：
  - [投票集成 Voting Ensemble](voting_ensemble.md)
  - [堆叠 Stacking](stacking.md)
  - [Bagging](bagging.md)（week 7 会讲）
  - [Boosting](boosting.md)

| 方法 | 基模型 | 训练方式 | 主要降低 |
|---|---|---|---|
| Voting | 通常异质 | 并行、独立 | 方差 |
| Stacking | 通常异质 | 两层：基模型 + 元模型 | 偏差 + 方差 |
| Bagging | 同质 | 并行，bootstrap 采样 | 方差 |
| Boosting | 同质（弱学习器） | 串行，关注前一轮错误 | 偏差 |
