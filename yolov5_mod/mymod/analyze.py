import os, re, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.transforms as mtrans
from matplotlib.font_manager import FontProperties
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    validation_curve,
)
from sklearn.ensemble import RandomForestClassifier
import mymod as mod
from myRandomForest import myRandomForestClassifier as myRandomForestClassifier
from pathlib import Path


def plot_bar_chart(Y_oringin, y_pred_all, ax, comfort_levels):
    # 把Y_oringin和y_pred_all输出到CSV文件
    data = pd.DataFrame({"Y_oringin": Y_oringin, "y_pred_all": y_pred_all})
    data.to_csv("analyze/data_y.csv", index=False)

    # 统计每个(区域号, 条形号)对应的个数
    counts = np.zeros((7, 7), dtype=int)
    for i in range(len(Y_oringin)):
        pred_val = int(y_pred_all[i] + 3)  # 区域号从-3到3，转换成列表索引0到6
        origin_val = int(Y_oringin[i] + 3)  # 条形号从-3到3，转换成列表索引0到6
        counts[pred_val, origin_val] += 1  # 对应区域号和条形号的个数加1
    # print(f"counts:\n{counts}")

    # 绘制条形图
    # fig, ax = plt.subplots()
    x = np.arange(7)
    width = 0.7  # 条形的宽度

    color_map = cm.rainbow(np.linspace(0, 1, 7))  # 生成彩虹色调色板

    for i in range(7):  # 遍历条形号
        ax.bar(
            x + (i - 3) * width / 7,
            counts[i],
            width / 7,
            label=comfort_levels[i],
            color=color_map[i],
        )

    # ax.set_xlabel("Bar Number")
    ax.set_ylabel("Counts")
    ax.set_title("Counts by Zone and Bar Number")
    ax.set_xticks(x)
    # ax.set_xticklabels(comfort_levels)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))  # 将图例显示在图表外部


def svm_demo(X_train, X_test, Y_train, Y_test, X1_name, X2_name):
    # clf = SVC(kernel="rbf", C=80)
    # clf.fit(X_train, Y_train)
    clf = SVC(kernel="rbf")
    # 定义要尝试的C值
    param_grid = {"C": range(1, 101)}
    grid_search = GridSearchCV(clf, param_grid, cv=5)
    grid_search.fit(X_train, Y_train)
    # print("Best parameters: ", grid_search.best_params_)
    # print("Best score: ", grid_search.best_score_)
    clf = SVC(kernel="rbf", C=grid_search.best_params_["C"])
    clf = grid_search.best_estimator_
    # clf.fit(X_train, Y_train)

    # 预测分类结果
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # 计算分类结果的准确率
    accuracy_train = np.round(accuracy_score(Y_train, y_pred_train), 2)
    accuracy_test = np.round(accuracy_score(Y_test, y_pred_test), 2)

    # # 分别合并train和test的X与Y数据
    # X_all = np.concatenate((X_train, X_test))
    # y_pred_all = np.concatenate((y_pred_train, y_pred_test))

    return y_pred_train, y_pred_test, accuracy_train, accuracy_test


def val_curve_demo(data_temper, X1_name, X2_name, data_TSV):
    # 以temperatur_data为输入，输出sensation的分类结果
    # 训练SVM分类器
    X = data_temper[[X1_name, X2_name]].values
    Y = data_TSV.values
    params = {
        "n_estimators": [200],
        "max_depth": [4],
        "min_samples_split": [10],
        "min_samples_leaf": [10],
        # "max_features": ["sqrt", "log2", None],
    }
    param = {"name": "max_depth", "range": range(1, 10)}
    train_scores, test_scores = validation_curve(
        RandomForestClassifier(
            n_estimators=100, min_samples_split=45, min_samples_leaf=15
        ),
        X,
        Y,
        param_name=param["name"],
        param_range=param["range"],
        cv=5,
        scoring="accuracy",
    )

    plt.figure()
    plt.plot(param["range"], np.mean(train_scores, axis=1), label="Training score")
    plt.plot(
        param["range"], np.mean(test_scores, axis=1), label="Cross-validation score"
    )
    plt.xlabel(param["name"])
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()


def randomForest_demo(
    X_train, X_test, Y_train, Y_test, X1_name, X2_name, isSeach: bool = False
):
    if isSeach:
        params_search = {
            "n_estimators": [1, 2, 5, 10, 20, 50, 100],
            "max_depth": [None, 1, 2, 3, 4, 5, 10],
            "min_samples_leaf": [1, 2, 5, 10],
            "min_samples_split": [2, 5, 10],
            "random_state": [42],
            # "max_features": ["sqrt", "log2", None],
        }

        # model_search = RandomizedSearchCV(RandomForestClassifier(), params_search, n_iter=10, cv=5)
        model_search = GridSearchCV(
            RandomForestClassifier(), params_search, cv=5, n_jobs=5, scoring="f1_macro"
        )
        model_search.fit(X_train, Y_train)
        cv_results = model_search.cv_results_.copy()
        best_params = model_search.best_params_.copy()
        best_score = model_search.best_score_
        print("Best parameters: ", best_params)
        print("Best score: ", best_score)
        mod.write_to_csv(Path(r"analyze\csv\data_rf.csv"), cv_results)
    else:
        best_params = {
            "n_estimators": 5,
            "max_depth": 5,
            "min_samples_leaf": 1,
            "min_samples_split": 5,
            "random_state": 42,
        }
    rf_model = RandomForestClassifier(**best_params)
    rf_model.fit(X_train, Y_train)  # 训练模型

    # mod.exportTree(rf_model, X1_name, X2_name)

    # return
    # rf_model = model_search.best_estimator_

    # 预测分类结果
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)

    # 计算分类结果的准确率
    accuracy_train = np.round(accuracy_score(Y_train, y_pred_train), 2)
    accuracy_test = np.round(accuracy_score(Y_test, y_pred_test), 2)

    # # 分别合并train和test的X与Y数据
    # X_all = np.concatenate((X_train, X_test))
    # y_pred_all = np.concatenate((y_pred_train, y_pred_test))

    best_params_copy = best_params.copy()
    best_params_copy.update(
        {
            "accuracy_test": accuracy_test,
            "accuracy_train": accuracy_train,
            "X1_name": X1_name,
            "X2_name": X2_name,
        }
    )

    # 导出模型
    joblib.dump(
        rf_model,
        rf"analyze/models/random_forest_model_{accuracy_test}_{X1_name}_{X2_name}.pkl",
    )

    mod.write_to_csv(Path(r"analyze\csv\data_rf_train.csv"), best_params_copy)

    return (
        y_pred_train,
        y_pred_test,
        accuracy_train,
        accuracy_test,
        best_params,
    )


def train(data_temper, X1_name, X2_name, data_TSV):
    feature_names = [X1_name, X2_name, "env"]
    X = data_temper[feature_names].values
    Y = data_TSV.values
    comfort_levels = [
        "冷",
        "凉",
        "微凉",
        "中性",
        "微暖",
        "暖",
        "热",
    ]
    dir_img = mod.getDir(r"analyze\img\randomForest\exp")

    # # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    (
        y_pred_train,
        y_pred_test,
        accuracy_train,
        accuracy_test,
        best_params,
    ) = randomForest_demo(
        X_train, X_test, Y_train, Y_test, X1_name, X2_name, isSeach=False
    )
    # randomForest_demo(X_train, X_test, Y_train, Y_test, X1_name, X2_name, isSeach=False)
    # return
    print(f"Accuracy of train ({X1_name}, {X2_name}): {accuracy_train}")
    print(f"Accuracy of test ({X1_name}, {X2_name}): {accuracy_test}")

    # # 分别合并train和test的X与Y数据
    # X_all = np.concatenate((X_train, X_test))
    # y_pred_all = np.concatenate((y_pred_train, y_pred_test))

    # 创建子图布局为2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    # fig.tight_layout()
    config_plt = {
        "font.family": "SimSun, Times New Roman",  # 使用衬线体
        "font.serif": ["SimSun"],  # 全局默认使用衬线宋体
        "font.sans-serif": ["SimSun"],  # 全局默认使用衬线宋体
        # "font.size": 14,  # 五号，10.5磅
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
    }
    plt.rcParams.update(config_plt)

    # 计算训练集的混淆矩阵
    conf_matrix_train = np.round(
        confusion_matrix(y_pred_train, Y_train, normalize="pred"), 2
    )
    # print(f"Confusion matrix of train ({X1_name}, {X2_name}):\n{conf_matrix_train}")

    # 绘制热力图
    heatmap = sns.heatmap(
        conf_matrix_train,
        annot=True,
        cmap="Blues",
        fmt="g",
        # xticklabels=clf.classes_,
        # yticklabels=clf.classes_,
        ax=ax1,
    )
    heatmap.set_xticklabels(comfort_levels)
    heatmap.set_yticklabels(comfort_levels)
    # for label in ax1.xaxis.get_ticklabels():
    #     label.set_fontname("SimSun")
    # for label in ax1.yaxis.get_ticklabels():
    #     label.set_fontname("SimSun")
    ax1.set_xlabel("预测分类")
    ax1.set_ylabel("实际分类")
    ax1.set_title("训练集混淆矩阵")

    plot_bar_chart(Y_train, y_pred_train, ax2, comfort_levels)
    # ax2.set_ylim(0, 16)
    ax2.set_xticklabels(comfort_levels)
    ax2.set_xlabel("预测分类")
    ax2.set_ylabel("实际分类计数")
    ax2.set_title("训练集预测分类中的实际分类分布")

    # 计算测试集的混淆矩阵
    conf_matrix_test = np.round(
        confusion_matrix(y_pred_test, Y_test, normalize="pred"), 2
    )
    # print(f"Confusion matrix of test ({X1_name}, {X2_name}):\n{conf_matrix_test}")

    # 绘制热力图
    heatmap = sns.heatmap(
        conf_matrix_test,
        annot=True,
        cmap="Blues",
        fmt="g",
        # xticklabels=clf.classes_,
        # yticklabels=clf.classes_,
        ax=ax3,
    )
    heatmap.set_xticklabels(comfort_levels)
    heatmap.set_yticklabels(comfort_levels)
    # for label in ax3.xaxis.get_ticklabels():
    #     label.set_fontname("SimSun")
    # for label in ax3.yaxis.get_ticklabels():
    #     label.set_fontname("SimSun")
    ax3.set_xlabel("预测分类")
    ax3.set_ylabel("实际分类")
    ax3.set_title("测试集混淆矩阵")

    plot_bar_chart(Y_test, y_pred_test, ax4, comfort_levels)
    # ax4.set_ylim(0, 4.5)
    ax4.set_xticklabels(comfort_levels)
    ax4.set_xlabel("预测分类")
    ax4.set_ylabel("实际分类计数")
    ax4.set_title("测试集预测分类中的实际分类分布")

    # plt.colorbar()
    # 在整张图的顶部添加标题
    subtitle = fig.suptitle(
        f"特征: {feature_names};\t\t训练准确度: {accuracy_train};\t\t测试准确度: {accuracy_test}\n超参数: {best_params}",
        x=0.1,
    )
    subtitle.set_ha("left")

    # 例遍fig中的子图，导出图片
    for ax in fig.axes:
        filename_img = rf"{dir_img}\{accuracy_test:.2f}_{X1_name}_{X2_name}\{accuracy_test:.2f}_{X1_name}_{X2_name}_{ax.get_title().replace(' ', '_')}.png"
        # filename_img的文件夹不存在则创建
        if not os.path.exists(os.path.dirname(filename_img)):
            os.makedirs(os.path.dirname(filename_img))
        mod.save_subfig(ax, filename_img, dpi=300)

    # 导出图片
    filename_img = rf"{dir_img}\{accuracy_test:.2f}_{X1_name}_{X2_name}.png"
    plt.savefig(
        filename_img,
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()


def test(data_temper, X1_name, X2_name, data_TSV):
    feature_names = [X1_name, X2_name, "env"]
    X = data_temper[feature_names].values
    Y = data_TSV.values

    # # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    best_params = {
        "n_estimators": 5,
        "max_depth": 5,
        "min_samples_leaf": 1,
        "min_samples_split": 5,
        "random_state": 42,
    }
    model = myRandomForestClassifier(
        n_trees=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        min_samples_split=best_params["min_samples_split"],
        random_state=best_params["random_state"],
    )
    model.fit(X_train, Y_train)  # 训练模型

    # 预测分类结果
    y_pred_train = model.predict(X_train)
    # print(f"y_pred_train:\n{y_pred_train}")
    y_pred_test = model.predict(X_test)

    # 计算分类结果的准确率
    # accuracy_train = np.round(accuracy_score(Y_train, y_pred_train), 2)
    # accuracy_test = np.round(accuracy_score(Y_test, y_pred_test), 2)
    accuracy_train = np.round(model.score(Y_train, y_pred_train), 2)
    accuracy_test = np.round(model.score(Y_test, y_pred_test), 2)

    print(f"Accuracy of train ({X1_name}, {X2_name}): {accuracy_train}")
    print(f"Accuracy of test ({X1_name}, {X2_name}): {accuracy_test}")

    # 绘制混淆矩阵
    comfort_levels = [
        "冷",
        "凉",
        "微凉",
        "中性",
        "微暖",
        "暖",
        "热",
    ]
    conf_matrix_train = np.round(
        confusion_matrix(y_pred_test, Y_test, normalize="pred"), 2
    )
    # print(f"Confusion matrix of train ({X1_name}, {X2_name}):\n{conf_matrix_train}")

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(8, 8))
    config_plt = {
        "font.family": "SimSun, Times New Roman",  # 使用衬线体
        "font.serif": ["SimSun"],  # 全局默认使用衬线宋体
        "font.sans-serif": ["SimSun"],  # 全局默认使用衬线宋体
        # "font.size": 14,  # 五号，10.5磅
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
    }
    plt.rcParams.update(config_plt)
    heatmap = sns.heatmap(
        conf_matrix_train,
        annot=True,
        cmap="Blues",
        fmt="g",
        # xticklabels=clf.classes_,
        # yticklabels=clf.classes_,
        ax=ax,
    )
    heatmap.set_xticklabels(comfort_levels)
    heatmap.set_yticklabels(comfort_levels)
    ax.set_xlabel("预测分类")
    ax.set_ylabel("实际分类")
    ax.set_title("训练集混淆矩阵")
    # plt.colorbar()
    # 在整张图的顶部添加标题
    subtitle = fig.suptitle(
        f"特征: {feature_names};\t\t训练准确度: {accuracy_train};\t\t测试准确度: {accuracy_test}\n超参数: {best_params}",
        x=0.1,
    )
    subtitle.set_ha("left")
    plt.show()


def run():
    # 读取CSV文件并转换为Pandas DataFrame
    file_path = r"analyze\fulldata_proc.csv"
    data = pd.read_csv(file_path).dropna()
    # 数据按"Cheek"列的数值升序排序
    data = data.sort_values(by="Cheek")

    # 取Cheek,Nose,Mouth,Chin,Forehead列
    headers_temper = ["env", "Cheek", "Nose", "Mouth", "Chin", "Forehead"]
    data_temper = data[headers_temper]

    # 提取单元格值为0的行
    data_zero = data_temper[(data_temper == 0.0).any(axis=1)]
    print(f"data_zero:\n{data_zero}")
    # 从temperature_data中删除zero_data对应的行
    data_withoutZero = data.drop(data_zero.index)

    # 提取文件名列
    # img_names = data_temper_withoutZero["ID"]
    data_withoutZero.to_csv(r"analyze\data_withoutZero.csv", index=False)
    # 取出"TSV"列
    data_TSV = data_withoutZero["TSV"]
    data_temper_withoutZero = data_withoutZero[headers_temper]
    data_temper_withoutZero.to_csv(r"analyze\data_temper_withoutZero.csv", index=False)

    # 取temperatur_data的列headers进行两两组合生成数组
    headers = data_temper_withoutZero.drop("env", axis=1).columns
    combinations = [
        (headers[i], headers[j])
        for i in range(len(headers))
        for j in range(i + 1, len(headers))
    ]
    print(f"combinations: {combinations}")
    # # 删除文件夹analyze/img中的所有图片
    # for file in os.listdir(r"analyze\img"):
    #     if re.match(r".*\.png", file):
    #         os.remove(os.path.join(r"analyze\img", file))
    # 例遍combinations，调用svm_plot函数绘制不同特征组合的分类结果
    # for X_names in combinations:
    #     # print(f"X_names: {X_names}")
    #     train(data_temper_withoutZero, X_names[0], X_names[1], data_TSV)
    train(data_temper_withoutZero, "Cheek", "Nose", data_TSV)
    # val_curve_demo(data_temper_withoutZero, "Cheek", "Nose", data_TSV)
    # test(data_temper_withoutZero, "Cheek", "Nose", data_TSV)


if __name__ == "__main__":
    run()
