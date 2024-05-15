import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.transforms as mtrans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def csvIntegrate():
    # 读取CSV文件并转换为Pandas DataFrame
    src_dir = r"datasets\Charlotte-ThermalFace\csv.edit"  # \S1.edit.csv"
    target_path = r"analyze\Charlotte-ThermalFace\105.csv"

    # 获取文件夹中的所有csv文件
    file_list = [file for file in os.listdir(src_dir) if file.endswith(".csv")]

    # 创建一个空的DataFrame来存储所有CSV文件的数据
    all_data = pd.DataFrame()

    # 逐个读取CSV文件，并将其添加到all_data中
    for file in file_list:
        file_path = os.path.join(src_dir, file)
        data = pd.read_csv(file_path)
        if all_data.empty:
            all_data = pd.DataFrame(columns=data.columns)
        all_data = pd.concat([all_data, data])

    # all_data按id列升序排序
    all_data = all_data.sort_values(by="ID")

    # 将合并后的数据保存到新的CSV文件中
    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path))
    all_data.to_csv(target_path, index=False)

    # 获取all_data的"ID"列
    id_list = all_data["ID"].values
    src_dir = r"datasets\Charlotte-ThermalFace"
    for id in id_list:
        file_name = str(id) + ".jpg"
        img_dir = src_dir + "\\S" + re.findall(r"\d+(?=\d{4})", id)[0]
        img_path = img_dir + "\\" + file_name
        target_dir = r"analyze\Charlotte-ThermalFace"
        # 复制图片到目标文件夹中
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        target_path = os.path.join(target_dir, file_name)
        img_path = img_path.replace("/", "\\")
        target_path = target_path.replace("/", "\\")
        print(img_path, target_path)
        os.system(f"copy /y {img_path} {target_path}")


def csvOrganizer_Carlotte():
    src_path = r"results\detect\facialRegion\csv_tempercoefficient3\predictions.csv"
    sensation_path = r"analyze\Charlotte-ThermalFace\105.csv"
    target_path = r"analyze\predictions_Chartlotte_processed.csv"
    # 读取csv文件并删除最后一列
    data = pd.read_csv(src_path, header=None).iloc[:, :-1]
    data.columns = ["文件名", "区域", "温度"]

    # "文件名"列的值去除指定字符串".jpg"
    data["文件名"] = data["文件名"].str.replace(".jpg", "")
    # 将数据按文件名和区域分组，并计算每个组的平均温度
    grouped_data = data.groupby(["文件名", "区域"])["温度"].mean().round(3).unstack()
    # 自定义输出列的顺序
    desired_columns_order = ["文件名", "Cheek", "Nose", "Mouth", "Forehead", "Chin"]
    # 重置索引并填充缺失值为0
    result = (
        grouped_data.reset_index()
        .fillna(0)
        .reindex(columns=desired_columns_order, fill_value=0)
        .sort_values(by="文件名")
    )

    # 读取sensation_path文件
    sensation_data = pd.read_csv(sensation_path).sort_values(by="ID")
    result["sensation"] = sensation_data["Sensation"]

    # 保存整理后的数据到新的csv文件，编码为utf-8 with BOM
    # result.to_csv(r"analyze\predictions_processed.test1.csv", index=False, encoding="utf_8_sig")
    result.to_csv(target_path, index=False)


def csvOrganizer():
    src_path = r"results\detect\facialRegion\csv_tempercoefficient3\predictions.csv"
    target_path = r"analyze\predictions_Chartlotte_processed.csv"
    # 读取csv文件并删除最后一列
    data = pd.read_csv(src_path, header=None).iloc[:, :-1]
    data.columns = ["文件名", "区域", "温度"]

    # "文件名"列的值去除指定字符串".jpg"
    data["文件名"] = data["文件名"].str.replace(".jpg", "")
    # 将数据按文件名和区域分组，并计算每个组的平均温度
    grouped_data = data.groupby(["文件名", "区域"])["温度"].mean().round(3).unstack()
    # 自定义输出列的顺序
    desired_columns_order = ["文件名", "Cheek", "Nose", "Mouth", "Forehead", "Chin"]
    # 重置索引并填充缺失值为0
    result = (
        grouped_data.reset_index()
        .fillna(0)
        .reindex(columns=desired_columns_order, fill_value=0)
    )

    # 保存整理后的数据到新的csv文件，编码为utf-8 with BOM
    # result.to_csv(r"analyze\predictions_processed.test1.csv", index=False, encoding="utf_8_sig")
    result.to_csv(target_path, index=False)


def svm_plot(temperature_data, X1_name, X2_name, sen_data):
    # 创建子图布局为2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # X1_name = "Cheek"
    # X2_name = "Nose"

    X1_oringin = temperature_data[X1_name].values
    X2_oringin = temperature_data[X2_name].values
    Y_oringin = sen_data.values
    # 绘制原始数据散点图
    scatter = ax1.scatter(X1_oringin, X2_oringin, c=Y_oringin, cmap=cm.rainbow)
    ax1.set_xlabel(f"{X1_name} Temperature")
    ax1.set_ylabel(f"{X2_name} Temperature")
    ax1.set_title("Original Data")
    ax1.legend(handles=scatter.legend_elements()[0], labels=["-1", "0", "1"])
    fig.colorbar(scatter, ax=ax1)

    # 以temperatur_data为输入，输出sensation的分类结果
    # 训练SVM分类器
    X = temperature_data.values
    Y = sen_data.values

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    clf = SVC(kernel="rbf", C=2)
    clf.fit(X_train, Y_train)

    # 预测分类结果
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # 计算分类结果的准确率
    accuracy_test = accuracy_score(Y_test, y_pred_test)
    accuracy_train = accuracy_score(Y_train, y_pred_train)
    print("Accuracy of train:", accuracy_train)
    print("Accuracy of test:", accuracy_test)

    # 分别合并train和test的X与Y数据
    X_all = np.concatenate((X_train, X_test))
    y_pred_all = np.concatenate((y_pred_train, y_pred_test))

    # 绘制train和test结果的散点图到同一张图中
    scatter = ax2.scatter(X_all[:, 0], X_all[:, 1], c=y_pred_all, cmap=cm.rainbow)
    # scatter = ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train, cmap=cm.rainbow)
    # scatter = ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap=cm.rainbow)
    ax2.set_xlabel("X1")
    ax2.set_ylabel("X2")
    ax2.set_title("Training and Testing Result")
    fig.colorbar(scatter, ax=ax2)

    scatter = ax3.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train, cmap=cm.rainbow)
    ax3.set_xlabel("X1")
    ax3.set_ylabel("X2")
    ax3.set_title("Training result")
    fig.colorbar(scatter, ax=ax3)

    scatter = ax4.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap=cm.rainbow)
    ax4.set_xlabel("X1")
    ax4.set_ylabel("X2")
    ax4.set_title("Testing result")
    fig.colorbar(scatter, ax=ax4)

    # plt.colorbar()
    plt.show()


def svm_demo():
    # 读取CSV文件并转换为Pandas DataFrame
    file_path = r"analyze\predictions_Chartlotte_processed.csv"
    data = pd.read_csv(file_path)
    # 数据按"Cheek"列的数值升序排序
    data = data.sort_values(by="Cheek")

    # 若"sensation"列的值为2，则去除该行
    data = data[data["sensation"] != 2]
    # 去除0值行
    # 剔除data的"sensation"列
    temperature_data_withoutSen = data.drop("sensation", axis=1)
    # 提取单元格值为0的行
    zero_data = temperature_data_withoutSen[
        (temperature_data_withoutSen == 0.0).any(axis=1)
    ]
    print(zero_data)
    # 从temperature_data中删除zero_data对应的行
    temperature_data_withoutZero = data.drop(zero_data.index)

    # 提取文件名列
    img_names = temperature_data_withoutZero["ID"]
    temperature_data_withoutZero.to_csv(
        r"analyze\temperature_data.withName.csv", index=False
    )
    temperature_data_withoutZero.drop("ID", axis=1).to_csv(
        r"analyze\temperature_data.withSen.csv", index=False
    )
    # 取出"sensation"列
    sen_data = temperature_data_withoutZero["sensation"]
    temperature_data = temperature_data_withoutZero.drop("ID", axis=1).drop(
        "sensation", axis=1
    )
    temperature_data.to_csv(r"analyze\temperature_data.csv", index=False)

    # 取temperatur_data的列headers进行两两组合生成数组
    headers = temperature_data.columns
    combinations = [
        (headers[i], headers[j])
        for i in range(len(headers))
        for j in range(i + 1, len(headers))
    ]
    print(combinations)
    # 例遍combinations，调用svm_plot函数绘制不同特征组合的分类结果
    for item in combinations:
        svm_plot(temperature_data, item[0], item[1], sen_data)


def boxPlot_Carlotte():
    # 读取CSV文件并转换为Pandas DataFrame
    file_path = r"analyze\predictions_Chartlotte_processed.csv"
    data = pd.read_csv(file_path)
    # 数据按"Cheek"列的数值升序排序
    data = data.sort_values(by="Cheek")

    # 去除0值行
    # 剔除data的"文件名"和s"ensation"列
    temperature_data_withoutSen = data.drop("sensation", axis=1)
    # 提取单元格值为0的行
    zero_data = temperature_data_withoutSen[
        (temperature_data_withoutSen == 0.0).any(axis=1)
    ]
    print(zero_data)
    # 从temperature_data中删除zero_data对应的行
    temperature_data_withoutZero = data.drop(zero_data.index)

    # 提取文件名列
    img_names = temperature_data_withoutZero["文件名"]
    count_img_names = len(img_names)
    print(count_img_names)
    colors = cm.rainbow(np.linspace(0, 1, count_img_names))
    temperature_data_withoutZero.to_csv(
        r"analyze\temperature_data.withName.csv", index=False
    )
    temperature_data_withoutZero.drop("文件名", axis=1).to_csv(
        r"analyze\temperature_data.withSen.csv", index=False
    )

    temperature_data = temperature_data_withoutZero.drop("文件名", axis=1).drop(
        "sensation", axis=1
    )
    temperature_data.to_csv(r"analyze\temperature_data.csv", index=False)

    # return(0)

    # # 绘制箱线图
    # plt.figure(figsize=(10, 6))
    # temperature_data.boxplot()
    # plt.title("Temperature Distribution Across Facial Regions")
    # plt.xlabel("Facial Regions")
    # plt.ylabel("Temperature (Celsius)")
    # plt.show()

    # 创建子图，布局为2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    # 创建并训练PCA模型
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(temperature_data)
    # print(data_pca)
    # 获取转换后的两个分量
    component1 = data_pca[:, 0]
    component2 = data_pca[:, 1]
    print(len(component1), len(component2))
    # 绘制二维散点图
    ax1.scatter(component1, component2, color=colors)
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    ax1.set_title("Scatter Plot of 2D PCA Components")
    # 添加文件名标签
    for i, file_name in enumerate(img_names):
        # print(file_name)
        # 计算偏移量
        dx = 5  # 0.1 * (ax1.get_xlim()[1] - ax1.get_xlim()[0]) # x轴长度的1%作为偏移量
        dy = 0  # 0.1 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]) # y轴长度的1%作为偏移量
        offset = mtrans.offset_copy(
            ax1.transData, fig=ax1.figure, x=dx, y=dy, units="points"
        )
        ax1.text(
            component1[i],
            component2[i],
            file_name,
            fontsize=8,
            ha="left",
            color=colors[i],
            transform=offset,
        )

    # 使用t-SNE进行数据降维
    tsne = TSNE(n_components=2, perplexity=25, learning_rate=200, random_state=42)
    data_embedded = tsne.fit_transform(temperature_data)
    # 获取转换后的两个分量
    component1 = data_embedded[:, 0]
    component2 = data_embedded[:, 1]
    # 绘制二维散点图
    # plt.figure(figsize=(8, 6))
    ax2.scatter(component1, component2, color=colors)
    ax2.set_title("2D Visualization of Sample Data")
    ax2.set_xlabel("TSNE Component 1")
    ax2.set_ylabel("TSNE Component 2")
    # 添加文件名标签
    for i, file_name in enumerate(img_names):
        # 计算偏移量
        dx = 5  # 0.1 * (ax2.get_xlim()[1] - ax2.get_xlim()[0]) # x轴长度的1%作为偏移量
        dy = 0  # 0.1 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]) # y轴长度的1%作为偏移量
        offset = mtrans.offset_copy(
            ax2.transData, fig=ax2.figure, x=dx, y=dy, units="points"
        )
        ax2.text(
            component1[i],
            component2[i],
            file_name,
            fontsize=8,
            ha="left",
            color=colors[i],
            transform=offset,
        )

    # 提取特定区域温度数据
    region1 = "Cheek"
    region2 = "Nose"
    region_data = temperature_data[[region1, region2]].values
    # component1 = region_data[region1]
    # component2 = region_data[region2]
    component1 = region_data[:, 0]
    component2 = region_data[:, 1]
    # 绘制二维散点图
    # plt.figure(figsize=(8, 6))
    ax3.scatter(component1, component2, color=colors)
    ax3.set_title("2D Scatter Plot of {} vs {}".format(region1, region2))
    ax3.set_xlabel(region1)
    ax3.set_ylabel(region2)
    # 添加文件名标签
    # for i, file_name in enumerate(img_names):
    #     ax3.text(
    #         region_data.iloc[i][region1],
    #         region_data.iloc[i][region2],
    #         file_name,
    #         fontsize=8,
    #         ha="left",
    #         color=colors[i],
    #     )
    for i, file_name in enumerate(img_names):
        # 计算偏移量
        dx = 5  # 0.1 * (ax3.get_xlim()[1] - ax3.get_xlim()[0]) # x轴长度的1%作为偏移量
        dy = 0  # 0.1 * (ax3.get_ylim()[1] - ax3.get_ylim()[0]) # y轴长度的1%作为偏移量
        offset = mtrans.offset_copy(
            ax3.transData, fig=ax3.figure, x=dx, y=dy, units="points"
        )
        ax3.text(
            component1[i],
            component2[i],
            file_name,
            fontsize=8,
            ha="left",
            color=colors[i],
            transform=offset,
        )

    plt.show()


def boxPlot():
    # 读取CSV文件并转换为Pandas DataFrame
    file_path = r"analyze\predictions_Chartlotte_processed.csv"
    data = pd.read_csv(file_path)
    # 数据按"Cheek"列的数值升序排序
    data = data.sort_values(by="Cheek")
    # 剔除"Forehead"值为0的行
    data = data[data["Forehead"] != 0]
    img_names = data["文件名"]
    count_img_names = len(img_names)
    colors = cm.rainbow(np.linspace(0, 1, count_img_names))

    # 提取区域温度数据
    temperature_data = data.drop("文件名", axis=1)

    # # 绘制箱线图
    # plt.figure(figsize=(10, 6))
    # temperature_data.boxplot()
    # plt.title("Temperature Distribution Across Facial Regions")
    # plt.xlabel("Facial Regions")
    # plt.ylabel("Temperature (Celsius)")
    # plt.show()

    # 创建子图，布局为2x2
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))

    # 创建并训练PCA模型
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(temperature_data)
    # print(data_pca)
    # 获取转换后的两个分量
    component1 = data_pca[:, 0]
    component2 = data_pca[:, 1]
    # 绘制二维散点图
    ax1.scatter(component1, component2, color=colors)
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    ax1.set_title("Scatter Plot of 2D PCA Components")
    # 添加文件名标签
    for i, file_name in enumerate(img_names):
        # print(file_name)
        # 计算偏移量
        dx = 5  # 0.1 * (ax1.get_xlim()[1] - ax1.get_xlim()[0]) # x轴长度的1%作为偏移量
        dy = 0  # 0.1 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]) # y轴长度的1%作为偏移量
        offset = mtrans.offset_copy(
            ax1.transData, fig=ax1.figure, x=dx, y=dy, units="points"
        )
        ax1.text(
            component1[i],
            component2[i],
            file_name,
            fontsize=8,
            ha="left",
            color=colors[i],
            transform=offset,
        )

    # 使用t-SNE进行数据降维
    tsne = TSNE(n_components=2, perplexity=25, learning_rate=200, random_state=42)
    data_embedded = tsne.fit_transform(temperature_data)
    # 获取转换后的两个分量
    component1 = data_embedded[:, 0]
    component2 = data_embedded[:, 1]
    # 绘制二维散点图
    # plt.figure(figsize=(8, 6))
    ax2.scatter(component1, component2, color=colors)
    ax2.set_title("2D Visualization of Sample Data")
    ax2.set_xlabel("TSNE Component 1")
    ax2.set_ylabel("TSNE Component 2")
    # 添加文件名标签
    for i, file_name in enumerate(img_names):
        # 计算偏移量
        dx = 5  # 0.1 * (ax2.get_xlim()[1] - ax2.get_xlim()[0]) # x轴长度的1%作为偏移量
        dy = 0  # 0.1 * (ax2.get_ylim()[1] - ax2.get_ylim()[0]) # y轴长度的1%作为偏移量
        offset = mtrans.offset_copy(
            ax2.transData, fig=ax2.figure, x=dx, y=dy, units="points"
        )
        ax2.text(
            component1[i],
            component2[i],
            file_name,
            fontsize=8,
            ha="left",
            color=colors[i],
            transform=offset,
        )

    # 提取特定区域温度数据
    region1 = "Cheek"
    region2 = "Nose"
    region_data = data[[region1, region2]].values
    # component1 = region_data[region1]
    # component2 = region_data[region2]
    component1 = region_data[:, 0]
    component2 = region_data[:, 1]
    # 绘制二维散点图
    # plt.figure(figsize=(8, 6))
    ax3.scatter(component1, component2, color=colors)
    ax3.set_title("2D Scatter Plot of {} vs {}".format(region1, region2))
    ax3.set_xlabel(region1)
    ax3.set_ylabel(region2)
    # 添加文件名标签
    # for i, file_name in enumerate(img_names):
    #     ax3.text(
    #         region_data.iloc[i][region1],
    #         region_data.iloc[i][region2],
    #         file_name,
    #         fontsize=8,
    #         ha="left",
    #         color=colors[i],
    #     )
    for i, file_name in enumerate(img_names):
        # 计算偏移量
        dx = 5  # 0.1 * (ax3.get_xlim()[1] - ax3.get_xlim()[0]) # x轴长度的1%作为偏移量
        dy = 0  # 0.1 * (ax3.get_ylim()[1] - ax3.get_ylim()[0]) # y轴长度的1%作为偏移量
        offset = mtrans.offset_copy(
            ax3.transData, fig=ax3.figure, x=dx, y=dy, units="points"
        )
        ax3.text(
            component1[i],
            component2[i],
            file_name,
            fontsize=8,
            ha="left",
            color=colors[i],
            transform=offset,
        )

    plt.show()


if __name__ == "__main__":
    # csvIntegrate()
    # csvOrganizer_Carlotte()
    # boxPlot_Carlotte()
    svm_demo()
    pass
