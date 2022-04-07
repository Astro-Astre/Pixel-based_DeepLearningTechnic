import os
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from PIL import Image
import cv2
import pickle
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb


def cut(img):
    temp = img.reshape(-1)
    for i in range(temp.shape[0]):
        if temp[i] < 0:
            temp[i] = 0
        elif temp[i] > 255:
            temp[i] = 255
    return temp.reshape(3, 256, 256)


def load_img(filename):
    if ".fits" in filename:
        with fits.open(filename) as hdul:
            return hdul[0].data
    elif ".dat" in filename:
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        raise TypeError


def separate(raw, stretch, Q, kernel_size, threshold):
    lum = (raw[0] + raw[1] + raw[2]) / 3
    img = make_lupton_rgb(lum, lum, lum, stretch=stretch, Q=Q).reshape(-1)
    binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1].reshape(256, 256, 3)
    return cv2.medianBlur(binary_img, kernel_size)


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def do_separate(input_dir, output_dir):
    mkdir(output_dir)
    files = os.listdir(input_dir)
    for i in tqdm(range(len(files))):
        # for i in tqdm(range(1)):
        img = load_img(input_dir + files[i])
        if img.shape[0] == 3:
            mask = separate(img, Q=Q, stretch=0.08, kernel_size=13, threshold=20)[:, :, 0]
            output = np.concatenate((img, mask.reshape(1, 256, 256)), axis=0)
            with open(output_dir + files[i], 'wb') as f:
                pickle.dump(output, f)


def multi_mask(input_dir, output_dir):
    mkdir(output_dir)
    files = os.listdir(input_dir)
    for i in tqdm(range(len(files))):
        # for i in tqdm(range(1)):
        img = load_img(input_dir + files[i])
        if img.shape[0] == 3:
            mask = separate(img, Q=Q, stretch=0.08, kernel_size=13, threshold=20)[:, :, 0]
            mask[mask == 255] = 1
            output = img * mask
            with open(output_dir + files[i], 'wb') as f:
                pickle.dump(output, f)


if __name__ == '__main__':
    path = "/data/renhaoye/decals_2022/in_decals/fits/"
    stretch = 0.03
    Q = 10
    fits_dir = os.listdir(path)
    t = ["trainingSet", "testSet", "validationSet"]
    index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for k in range(len(t)):
        for i in range(len(index)):
            multi_mask("/data/renhaoye/decals_2022/in_decals/dataset/%s/%d/" % (t[k], index[i]),
                       "/data/renhaoye/decals_2022/in_decals/multi_mask_dataset/%s/%d/" % (t[k], index[i]))
