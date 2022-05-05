import csv
from preprocess.data_handle import *
from astropy.io import fits
import random
from functools import partial
import math
from tqdm import tqdm
import numpy as np
import multiprocessing

SAVE_PATH = "/data/renhaoye/decals_2022/"  # the head of the directory to save
PIXEL = 5
DEGREE = 30


class Img:
    def __init__(self, image, rows, cols, center=None):
        self.g_dst = None
        self.r_dst = None
        self.z_dst = None
        if center is None:
            center = [0, 0]
        self.dst = None
        self.g_src = image[0]
        self.r_src = image[1]
        self.z_src = image[2]
        self.transform = None
        self.rows = rows
        self.cols = cols
        self.center = center  # rotate center

    def Shift(self, delta_x, delta_y):  # 平移
        # delta_x>0 shift left  delta_y>0 shift top
        self.transform = np.array([[1, 0, delta_x],
                                   [0, 1, delta_y],
                                   [0, 0, 1]])

    def Flip(self):  # vertically flip
        self.transform = np.array([[-1, 0, self.rows - 1],
                                   [0, 1, 0],
                                   [0, 0, 1]])

    def Rotate(self, beta):  # rotate
        # beta<0 rotate clockwise
        self.transform = np.array([[math.cos(beta), -math.sin(beta), 0],
                                   [math.sin(beta), math.cos(beta), 0],
                                   [0, 0, 1]])

    def Process(self):
        self.g_dst = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.r_dst = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.z_dst = np.zeros((self.rows, self.cols), dtype=np.float32)
        for i in range(self.rows):
            for j in range(self.cols):
                src_pos = np.array([i - self.center[0], j - self.center[1], 1])
                [x, y, z] = np.dot(self.transform, src_pos)
                x = int(x) + self.center[0]
                y = int(y) + self.center[1]
                if x >= self.rows or y >= self.cols:
                    self.g_dst[i][j] = 0.
                    self.r_dst[i][j] = 0.
                    self.z_dst[i][j] = 0.
                else:
                    self.g_dst[i][j] = self.g_src[int(x)][int(y)]
                    self.r_dst[i][j] = self.r_src[int(x)][int(y)]
                    self.z_dst[i][j] = self.z_src[int(x)][int(y)]
        self.dst = np.array((self.g_dst, self.r_dst, self.z_dst))


def flip(img, save_dir):
    height, width = img.shape[1:3]
    output = Img(img, height, width, [0, 0])
    output.Flip()
    output.Process()
    save_fits(output.dst, save_dir + "_flipped.fits")


def rotate(img, save_dir):
    seed = random.randint(-DEGREE, DEGREE)
    height, width = img.shape[1:3]
    output = Img(img, height, width, [height / 2, width / 2])
    output.Rotate(seed)
    output.Process()
    save_fits(output.dst, save_dir + "_rotated.fits")


def shift(img, save_dir, pixel):
    height, width = img.shape[1:3]
    output = Img(img, height, width, [0, 0])
    output.Shift(pixel, 0)
    output.Process()
    save_fits(output.dst, save_dir + "_shifted.fits")


def augmentation(i, row):
    dst_dir = "/data/renhaoye/decals_2022/sdss_in_decals/agmtn/"  # 目标存放文件夹
    src_dir = "/data/renhaoye/sdss_dr7_decals_overlap/scaled/"  # 原始路径
    src_name = src_dir + row[i]  # 原始图片绝对路径
    save_name = dst_dir + row[i].split(".fits")[0]  # 保存绝对路径 不带扩展名
    if not os.path.exists(save_name + ".fits"):
        hdul = fits.open(src_name)
        scaled = hdul[0].data
        save_fits(scaled, save_name + '.fits')
        flip(scaled, save_name)
        shift(scaled, save_name, PIXEL)
        rotate(scaled, save_name)
        hdul.close()


if __name__ == "__main__":
    src_dir = "/data/renhaoye/sdss_dr7_decals_overlap/scaled/"  # 原始路径
    src_files = os.listdir(src_dir)
    print("start")
    index = []
    for i in range(len(src_files)):
        index.append(i)
    p = multiprocessing.Pool(15)
    p.map(partial(augmentation, row=src_files), index)
    p.close()
    p.join()
