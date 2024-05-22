# 根据conf_matrix_train绘制混淆矩阵
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

# 设置字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def plot_confusion_matrix(
    data, classes, title="Confusion matrix", cmap=plt.cm.Blues
):
    
    config_plt = {
        "font.family": "SimSun, Times New Roman",  # 使用衬线体
        "font.serif": ["SimSun"],  # 全局默认使用衬线宋体
        "font.sans-serif": ["SimSun"],  # 全局默认使用衬线宋体
        "font.size": 12,  # 五号，10.5磅
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
    }
    plt.rcParams.update(config_plt)

    heatmap = sns.heatmap(
        data,
        cmap=cmap,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 12},
        xticklabels=classes,
        yticklabels=classes,
    )
    heatmap.set_xlabel("真实分类")
    heatmap.set_ylabel("预测分类")
    # heatmap.set_title(title)
    heatmap.get_figure().savefig(".temp\面部区域预测混淆矩阵.png",dpi=300)
    plt.show()


if __name__ == "__main__":
    conf_matrix_train = [
        [1.00, 0, 0, 0, 0, 0.02],
        [0, 1.00, 0, 0, 0, 0.26],
        [0, 0, 1.00, 0, 0, 0.12],
        [0, 0, 0, 0.96, 0, 0.30],
        [0, 0, 0, 0, 1.00, 0.30],
        [0, 0, 0, 0.04, 0, 0],
    ]
    # 绘制混淆矩阵
    plot_confusion_matrix(
        conf_matrix_train,
        classes=["Nose", "Mouth", "Cheek", "Forehead", "Chin", "Background"],
        # normalize=True,
        title="混淆矩阵",
    )
