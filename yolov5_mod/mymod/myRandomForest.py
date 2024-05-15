import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import math


# 定义节点
class Node:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


# 定义决策树模型
class myDecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth  # 决策树深度
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    # 模型训练
    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    # 模型预测
    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    # 预测节点类型
    def _predict(self, inputs):
        node = self.tree_
        while node.value is None:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    # 生成树
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < 2:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_indices = range(n_features)
        best_feature, best_threshold = self._best_criteria(X, y, feature_indices)
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)

        left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    # 信息增益最大的特征和阈值
    def _best_criteria(self, X, y, feature_indices):
        best_gain = -1
        split_index, split_threshold = None, None
        for i in feature_indices:
            column = X[:, i]
            thresholds = set(column)
            for threshold in thresholds:
                gain = self._information_gain(y, column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_index = i
                    split_threshold = threshold
        return split_index, split_threshold

    # 信息增益
    def _information_gain(self, y, X_column, split_threshold):
        parent_entropy = self._entropy(y)
        left_indices, right_indices = self._split(X_column, split_threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        n = len(y)
        nl, nr = len(left_indices), len(right_indices)
        el = self._entropy(y[left_indices])
        er = self._entropy(y[right_indices])
        child_entropy = (nl / n) * el + (nr / n) * er
        ig = parent_entropy - child_entropy
        return ig

    # 熵
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / np.sum(hist)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    # 分割节点
    def _split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices

    # 返回标签
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    # 精度
    def score(self, y_pred, y):
        accuracy = (y_pred == y).sum() / len(y)
        return accuracy


class myRandomForestClassifier:
    def __init__(
        self,
        n_trees,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
    ):
        """
        :param n_trees: 随机森林包含的决策树数量
        :param max_depth: 决策树的最大深度
        :param min_samples_split: 分裂当前节点所需的当前节点所包含的最小样本数
        :param min_samples_leaf: 分裂当前节点所需的子节点所包含的最小样本数
        :param random_state: 随机种子，用于重现结果
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []

    # 模型训练
    def fit(self, X, y):
        """
        训练随机森林模型
        :param X: 训练集特征
        :param y: 训练集标签
        """
        np.random.seed(self.random_state)
        for i in range(self.n_trees):
            # 随机采样
            idx = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_sample = X[idx]
            y_sample = y[idx]

            # 构建决策树
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    # 模型预测
    def predict(self, X):
        """
        使用训练好的模型进行预测
        :param X: 测试集特征
        :return: 预测结果
        """
        y_pred = []
        for tree in self.trees:
            y_pred.append(tree.predict(X))
        y_pred = np.array(y_pred).T.astype(np.int64)
        y_pred_min = y_pred.min()
        if y_pred_min < 0:
            y_pred += abs(y_pred_min)
        result = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred
        )
        if y_pred_min < 0:
            result -= abs(y_pred_min)
        return result

    # 精度
    def score(self, y_pred, y):
        """
        计算分类的准确率
        :param y_pred: 预测值
        :param y: 真实值
        :return: 准确率
        """
        accuracy = (y_pred == y).sum() / len(y)
        return accuracy


if __name__ == "__main__":
    # 设置随机种子
    seed_value = 2023
    np.random.seed(seed_value)
    # 导入数据
    X, y = load_iris(return_X_y=True)
    X = X[:, :2]

    # 划分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed_value
    )

    # 训练随机森林
    model = myRandomForestClassifier(
        n_trees=100, max_depth=5, min_samples_split=2, random_state=seed_value
    )
    model.fit(X_train, y_train)

    # 结果
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    score_train = model.score(y_train_pred, y_train)
    score_test = model.score(y_test_pred, y_test)

    print("训练集Accuracy: ", score_train)
    print("测试集Accuracy: ", score_test)

    # # 可视化决策边界
    # x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    # x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    # Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    # Z = Z.reshape(xx1.shape)
    # plt.contourf(xx1, xx2, Z, cmap=plt.cm.Spectral)
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    # plt.xlabel("Sepal length")
    # plt.ylabel("Sepal width")
    # plt.savefig('a.png', dpi=720)
    # plt.show()
