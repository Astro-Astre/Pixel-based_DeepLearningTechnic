import csv
from data_handle import *
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
    save_dat(output.dst, save_dir + "_flipped.dat")


def rotate(img, save_dir):
    seed = random.randint(-DEGREE, DEGREE)
    height, width = img.shape[1:3]
    output = Img(img, height, width, [height / 2, width / 2])
    output.Rotate(seed)
    output.Process()
    save_dat(output.dst, save_dir + "_rotated.dat")


def shift(img, save_dir, pixel):
    height, width = img.shape[1:3]
    output = Img(img, height, width, [0, 0])
    output.Shift(pixel, 0)
    output.Process()
    save_dat(output.dst, save_dir + "_shifted.dat")


def augmentation(i, rows):
    label = rows[i][-1]
    path = rows[i][-2]
    agtn = rows[i][-1] in ["0", "4", "7", "8", "9"]
    save_dir = SAVE_PATH + "augmentation_auto/" + label + "/"
    mkdir(save_dir)
    save_name = save_dir + path.split("/")[-1].split(".fits")[0]
    if os.path.getsize(path) == 792000:
        hdul = fits.open(path)
        img, remove = normalization(hdul[0].data)
        hdul.close()
        if not remove:
            # saveImg(img, save_name + '_raw.dat')
            if agtn:
                # flip(img, save_name)
                # shift(img, save_name, PIXEL)
                rotate(img, save_name)


if __name__ == "__main__":
    with open("/data/renhaoye/Decals/label_auto_beforeAugmentation.csv", "r") as r:
        reader = csv.reader(r)
        title = next(reader)
        rows = [row for row in reader]
        for i in tqdm(range(len(rows))):
        # # for i in tqdm(range(1)):
            augmentation(i, rows)
        # index = []
        # for i in range(len(rows)):
        #     index.append(i)
        # p = multiprocessing.Pool(25)
        # p.map(partial(augmentation, rows=rows), index)
        # p.close()
        # p.join()
