# %%
import math
import os
import platform
from matplotlib import pyplot as plt
import cv2
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, T_co
from torchvision import transforms
from PIL import Image

env = platform.system().lower()
base_path = '../../' if env == 'windows' else '/home/yongxinge/liao/'
batch_size = 32 if env == 'windows' else 128
train_epochs = 8 if env == 'windows' else 100
learn_rate = 0.03

train_data_path = base_path + 'binaryTrainGamma/'
test_data_path = base_path + 'binaryTestGamma/'

patch_height = 100
patch_width = 100


def get_illum_mean_and_std(img):
    is_gray = img.ndim == 2 or img.shape[1] == 1
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 2] / 255
    return round(np.mean(illum), 4), round(np.std(illum), 4)


def gamma_transform(img, gamma):
    is_gray = img.ndim == 2 or img.shape[1] == 1
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 2] / 255.
    illum = np.power(illum, gamma)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 2] = v.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# %%
def gamma_trans(img):
    img_result = img
    mean_max, std_max = get_illum_mean_and_std(img=img)
    for i in np.arange(0.5, 2, 0.1):
        cur_img = gamma_transform(img=img, gamma=i)
        mean_cur, std_cur = get_illum_mean_and_std(img=cur_img)
        if std_cur > std_max and mean_cur >= mean_max:
            mean_max = mean_cur
            std_max = std_cur
            img_result = cur_img
    return img_result


# %%
def img_transform(img, cols=12, rows=9):
    height = img.shape[0]
    width = img.shape[1]
    print(height)
    print(width)
    col_size = int(width / cols)
    row_size = int(height / rows)
    for i in range(rows):
        for j in range(cols):
            patch = img[i * row_size:(i + 1) * row_size,
                    j * col_size:(j + 1) * col_size]
            patch = gamma_trans(patch)
            img[i * row_size:(i + 1) * row_size,
            j * col_size:(j + 1) * col_size] = patch
    return img


if __name__=='__main__':
    img_cv = cv2.imread("test.png")  # 读取数据
    img_cv = img_transform(img_cv)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    plt.imshow(bgr)
    cv2.imwrite("ruihua.jpg",bgr)
    plt.show()