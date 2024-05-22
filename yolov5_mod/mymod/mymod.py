import os, cv2, csv
import numpy as np
from sklearn import tree
import pydot, pydotplus
from io import StringIO
import matplotlib.pyplot as plt

def save_subfig(ax,path,dpi=100):
    bbox = ax.get_tightbbox(ax.figure.canvas.get_renderer()).expanded(1.02, 1.02)
    extent = bbox.transformed(ax.figure.dpi_scale_trans.inverted())
    ax.figure.savefig(path, bbox_inches=extent,dpi=dpi)

def getDir(dir):
    dir = r"analyze\img\randomForest\exp"

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        # 遍历目标文件夹，查找可用的文件夹名
        folder_index = 1
        while True:
            new_dir = f"{dir}{folder_index}"
            if not os.path.exists(new_dir):
                dir = new_dir
                os.makedirs(dir)
                break
            folder_index += 1
    return dir


def write_to_csv(csv_path, data: dict, is_header: bool = False):
    """Writes prediction data for an image to a CSV file, appending if the file exists."""
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        # if not csv_path.is_file():
        if f.tell() == 0 or is_header:
            writer.writeheader()
        writer.writerow(data)


def areaMeanTemper(img: cv2.typing.MatLike, xyxy: list):
    xy0 = np.array(xyxy[0:2]).astype(int)
    xy1 = np.array(xyxy[2:4]).astype(int)
    # 切割图像
    # img_area=img[y1:y2, x1:x2]
    img_area = img[xy0[1] : xy1[1], xy0[0] : xy1[0]]
    # 保存图像
    # cv2.imwrite("../.temp/img/temp_area.jpg", img)
    # 转换为灰度图
    img_gray = cv2.cvtColor(img_area, cv2.COLOR_BGR2GRAY)
    # 数组归一化
    array_img = np.array(img_gray) / 255
    # 求取平均值保留三位有效数字
    meanGrayValue = np.mean(array_img)
    temper0 = np.around(meanGrayValue * 10.843 + 24.797, decimals=3)
    temper1 = np.around(meanGrayValue * 25.531 + 12.738, decimals=3)
    unit = "degC"
    # return temper0
    return temper0


def exportTree(rf_model, X1_name, X2_name):
    # 选择要绘制的决策树索引，例如第一棵树
    tree_index = 0

    # 创建输出流
    dot_data = StringIO()
    # 导出决策树
    tree.export_graphviz(
        rf_model.estimators_[tree_index],
        out_file=dot_data,
        filled=True,
        rounded=True,
        special_characters=True,
        feature_names=[X1_name, X2_name, "env"],
    )
    # 使用pydot将导出的决策树绘制到图形中
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf(rf".temp\decision_tree_{tree_index}.pdf")
