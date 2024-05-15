import cv2
import csv
import numpy as np


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
