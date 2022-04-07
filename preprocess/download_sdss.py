from astropy.io import fits
import numpy as np
import requests
from tqdm import tqdm
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    for index in range(7):
        dir = os.listdir("/data/renhaoye/Decals/dataset_auto/validationSet/%d" % index)
        with open("/data/renhaoye/decals_2022/fits_all_%d.txt" % index, "w+") as f:
            for i in dir:
                # noinspection PyTypeChecker
                f.write(i.split("_raw.dat")[0].split("_")[0] + "," + i.split("_raw.dat")[0].split("_")[1] + "\n")

    # data = None
    # with open("download_t.txt", "r") as f:  # 打开文件
    #     data = f.read()  # 读取文件
    # data = data.split(' ')
    # # print(data)
    # for url in tqdm(data):
    #     filename = url.split('/')[-1]
    #     if os.path.exists('/Users/cmloveczy/Downloads/陈小宓上台项目/astro2/fits/' + filename) != True:
    #         response = requests.get(url)
    #         img = response.content
    #         with open('/Users/cmloveczy/Downloads/陈小宓上台项目/astro2/fits/' + filename, 'wb') as f:
    #              f.write(img)
    #         # print(url)
    #     else:
    #         print('/Users/cmloveczy/Downloads/陈小宓上台项目/astro2/fits/' + filename + '已经存在！')
