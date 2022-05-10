import os

from preprocess.data_handle import *
import pandas as pd
from functools import partial
from tqdm import tqdm
import multiprocessing

if __name__ == "__main__":
    decals_df = pd.read_csv("/data/renhaoye/decals_2022/dataset_decals.csv", index_col=0)
    # """
    # BEST
    # """
    # rows = []  # 增强的列表
    # agmtn = []  # 数据增强与否
    # for i in range(len(decals_df)):
    #     ra, dec = str(decals_df.iloc[i].ra), str(decals_df.iloc[i].dec)
    #     rows.append(ra + "_" + dec + ".fits")
    #     if decals_df.iloc[i].label == 0 or decals_df.iloc[i].label == 6:
    #         if decals_df.iloc[i].func == 'train':
    #             agmtn.append(True)  # 0和6的训练集要增强
    #         else:
    #             agmtn.append(False)
    #     else:
    #         agmtn.append(False)
    # index = []
    # for i in range(len(rows)):
    #     index.append(i)
    # p = multiprocessing.Pool(10)
    # p.map(partial(augmentation, rows=rows), index)
    # p.close()
    # p.join()

    # """
    # baseline
    # """
    # rows = []  # 增强的列表
    # agmtn = []  # 数据增强与否
    # for i in range(len(decals_df)):
    #     ra, dec = str(decals_df.iloc[i].ra), str(decals_df.iloc[i].dec)
    #     rows.append(ra + "_" + dec + ".fits")
    #     agmtn.append(False)
    # index = []
    # for i in range(len(rows)):
    #     index.append(i)
    # p = multiprocessing.Pool(10)
    # p.map(partial(augmentation, rows=rows, src_dir="in_decals/fits/", dst_dir="in_decals/decals_basline/", scale=False, agmtn=agmtn, gaia=False), index)
    # p.close()
    # p.join()

    # """
    # NoAgmtn
    # """
    # rows = []  # 增强的列表
    # agmtn = []  # 数据增强与否
    # for i in range(len(decals_df)):
    #     ra, dec = str(decals_df.iloc[i].ra), str(decals_df.iloc[i].dec)
    #     rows.append(ra + "_" + dec + ".fits")
    #     agmtn.append(False)
    # index = []
    # for i in range(len(rows)):
    #     index.append(i)
    # p = multiprocessing.Pool(10)
    # p.map(partial(augmentation, rows=rows, src_dir="in_decals/fits/", dst_dir="in_decals/decals_basline/", scale=True,
    #               agmtn=agmtn, gaia=False), index)
    # p.close()
    # p.join()

    """
    Masked
    """
    rows = []  # 增强的列表
    agmtn = []  # 数据增强与否
    for i in range(len(decals_df)):
        ra, dec = str(decals_df.iloc[i].ra), str(decals_df.iloc[i].dec)
        # print(os.path.exists(data_config.root_path + "in_decals/decals_masked/" + ra + "_" + dec + ".fits"))
        if not os.path.exists(data_config.root_path + "in_decals/decals_masked/" + ra + "_" + dec + ".fits"):
            rows.append(ra + "_" + dec + ".fits")
            if decals_df.iloc[i].label == 0 or decals_df.iloc[i].label == 6:
                if decals_df.iloc[i].func == 'train':
                    agmtn.append(True)  # 0和6的训练集要增强
                else:
                    agmtn.append(False)
            else:
                agmtn.append(False)
    index = []
    for i in range(len(rows)):
        index.append(i)
    p = multiprocessing.Pool(10)
    p.map(partial(augmentation, rows=rows, src_dir="in_decals/fits/", dst_dir="in_decals/decals_masked/", scale=True,
                  agmtn=agmtn, gaia=True), index)
    p.close()
    p.join()
