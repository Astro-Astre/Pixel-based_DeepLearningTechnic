import os
import numpy as np
import pandas as pd
from astropy.wcs import WCS
from astropy.io import fits
from tqdm import tqdm
import warnings
from functools import partial
import multiprocessing
from data_handle import *

SDSS = pd.read_csv("/data/renhaoye/decals_2022/sdss_in_decals.csv")


class Cut:
    def __init__(self, info):
        assert type(info) == pd.core.series.Series, "please input pd.core.series.Series like: df.iloc[index]"
        self.ra = info.ra
        self.dec = info.dec
        self.run = info.run
        self.rerun = info.rerun
        self.camcol = info.camcol
        self.field = info.field
        self.cutout = np.zeros((3, 256, 256))

    def __call__(self, *args, **kwargs):
        new_field = str(self.field)
        new_run = str(self.run)
        if len(new_run) == 3:
            new_run = "0" + new_run
        if len(new_field) == 2:
            new_field = "0" + new_field
        raw_path = "/data/renhaoye/sdss_dr7_decals_overlap/raw/"
        bands = ["g", "r", "z"]
        files = []
        for band in bands:
            files.append(raw_path + "fpC-00%s-%s%d-0%s.fit" % (new_run, band, self.camcol, new_field))
        for i in range(len(files)):
            with fits.open(files[i]) as hdul:
                data = hdul[0].data
                wcs = WCS(hdul[0].header)
                y, x = wcs.all_world2pix([[self.ra, self.dec]], 0)[0]
                if type(y) == np.float64 and type(x) == np.float64:
                    x, y = int(x), int(y)
                    left_upper_x, left_upper_y = x - 128, y - 128
                    if left_upper_x > 1233 or left_upper_x < 0 or (left_upper_y > 1790) or left_upper_y < 0:
                        self.cutout[i] = np.zeros((256, 256))
                    else:
                        self.cutout[i] = data[left_upper_x:left_upper_x + 256, left_upper_y:left_upper_y + 256]
                else:
                    self.cutout[i] = np.zeros((256, 256))


def write2fits(data: np.ndarray, filename: str):
    """
    将ndarray保存成fits文件
    :param data: 待保存数据
    :param filename: 保存文件名
    :return:
    """
    if len(data.shape) == 2:
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
    elif data.shape[-1] == 3:
        g, r, z = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        data = np.array((g, r, z))
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
    elif data.shape[0] == 3:
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename, overwrite=True)
        hdulist.close()
    else:
        raise RuntimeError


def do(i):
    raw_path = "/data/renhaoye/sdss_dr7_decals_overlap/test/"
    ra = str(SDSS.iloc[i].ra).split(".")[0] + "." + str(SDSS.iloc[i].ra).split(".")[1][0:6]
    dec = str(SDSS.iloc[i].dec).split(".")[0] + "." + str(SDSS.iloc[i].dec).split(".")[1][0:6]
    a = Cut(SDSS.iloc[i])
    a()
    if np.max(a.cutout[0, :, :]) != 0. and np.max(a.cutout[1, :, :]) != 0. and np.max(a.cutout[2, :, :]) != 0.:
        normalized_unlock = normalization(a.cutout / 65535, shape="CHW", lock=False)  # 先做一次分通道归一化
        if not type(normalized_unlock) == int:
            normalized_lock = normalization(normalized_unlock, shape="CHW", lock=True)  # 再做一次全通道归一化
            if not type(normalized_lock) == int:
                scaled = auto_scale(normalized_lock)  # 再做stf
                write2fits(scaled, "/data/renhaoye/sdss_dr7_decals_overlap/scaled/%s_%s.fits" % (ra, dec))
    with open("/data/renhaoye/sdss_dr7_decals_overlap/done_index.txt", "a+") as w:
        w.write(str(i) + "\n")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # for i in tqdm(range(len(SDSS))):
    # for i in tqdm(range(10)):
    #     do(i)
    index = []
    for i in range(len(SDSS)):
        index.append(i)
    exist = []
    with open("/data/renhaoye/sdss_dr7_decals_overlap/done_index.txt", "r") as r:
        for i in r:
            exist.append(i.split("\n")[0])
    with open("/data/renhaoye/sdss_dr7_decals_overlap/resume_index.txt", "w") as w:
        for e in tqdm(range(len(index))):
            if str(index[e]) not in exist:
                w.write(str(index[e]) + "\n")
    exist = []
    with open("/data/renhaoye/sdss_dr7_decals_overlap/resume_index.txt", "r") as r:
        for i in r:
            exist.append(int(i.split("\n")[0]))
    p = multiprocessing.Pool(20)
    p.map(partial(do), exist)
    p.close()
    p.join()
