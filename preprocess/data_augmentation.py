import csv
from preprocess.data_handle import *
from astropy.io import fits
import random
from functools import partial
import math
from tqdm import tqdm
import numpy as np
import multiprocessing
from data_handle import *

SAVE_PATH = "/data/renhaoye/decals_2022/"  # the head of the directory to save
PIXEL = 5
DEGREE = 30


def augmentation(i, rows, ):
    dst_dir = SAVE_PATH + "in_decals/augmentation_all/"  # 目标存放文件夹
    img_name = rows[i][1] + "_" + rows[i][2]  # 图像名：ra_dec
    src_dir = "/data/renhaoye/decals_2022/in_decals/fits/"  # 原始路径
    src_name = src_dir + img_name + ".fits"  # 原始图片绝对路径
    save_name = dst_dir + img_name  # 保存绝对路径 不带扩展名
    if not os.path.exists(save_name + ".fits"):
        hdul = fits.open(src_name)  # 打开fits文件
        normalized_unlock = normalization(hdul[0].data, shape="CHW", lock=False)  # 先做一次分通道归一化
        if not type(normalized_unlock) == int:
            normalized_lock = normalization(normalized_unlock, shape="CHW", lock=True)  # 再做一次全通道归一化
            if not type(normalized_lock) == int:
                scaled = auto_scale(normalized_lock)  # 再做stf
                hdul.close()
                save_fits(scaled, save_name + '.fits')
                flip(scaled, save_name)
                shift(scaled, save_name, PIXEL)
                rotate(scaled, save_name)
            else:
                os.system("mv %s /data/renhaoye/decals_2022/in_decals/fits_deleted/" % src_name)
        else:
            os.system("mv %s /data/renhaoye/decals_2022/in_decals/fits_deleted/" % src_name)


if __name__ == "__main__":
    with open("/data/renhaoye/decals_2022/fits.csv", "r") as r:
        reader = csv.reader(r)
        title = next(reader)
        rows = [row for row in reader]
        # print(rows[1])
        # # for i in tqdm(range(len(rows))):
        # for i in tqdm(range(1)):
        #     augmentation(i, rows)
        index = []
        for i in range(len(rows)):
            index.append(i)
        p = multiprocessing.Pool(20)
        p.map(partial(augmentation, rows=rows), index)
        p.close()
        p.join()
