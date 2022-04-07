import os
import pandas as pd
from pandas import read_parquet
from tqdm import tqdm
from functools import partial
import multiprocessing


def get_url(ra, dec):
    # ra = df.iloc[i, 0]
    # dec = df.iloc[i, 1]
    save_dir = "/data/renhaoye/decals_2022/in_decals/jpg/%f_%f.jpg" % (ra, dec)
    control = "https://www.legacysurvey.org/viewer/jpeg-cutout?ra=%f&dec=%f&layer=dr8&pixscale=0.262&bands=grz'" % (
        ra, dec)
    url = "wget '" + control + " -q -O " + save_dir
    # os.system(url)
    return url


def download_file(filename):
    os.system(filename)


if __name__ == '__main__':
    fits_path = "/data/renhaoye/decals_2022/in_decals/fits"  # 原始fits文件保存位置
    fits_list = os.listdir(fits_path)  # 获取fits文件列表
    poi_list = []  # 获取最后的ra，dec列表名(小数点后六位)
    for i in range(len(fits_list)):
        poi_list.append([float(fits_list[i].split("_")[0]), float(fits_list[i].split("_")[1].split(".fits")[0])])
    poi_list = pd.DataFrame(poi_list, columns=["ra", "dec"])

    auto_catalog_path = "/data/renhaoye/decals_2022/gz_decals_auto_posteriors.parquet"  # auto列表的位置
    df_auto = read_parquet(auto_catalog_path).rename(columns=lambda x: x.replace("-", "_"))  # 头中的-换为_
    df_auto = df_auto.loc[:, ["ra", "dec"]]
    # auto列表中的ra和dec保留小数点后六位
    resolution = 6
    df_auto.ra = df_auto.ra.map(lambda x: float(str(x).split(".")[0] + "." + str(x).split(".")[1][:resolution]))
    df_auto.dec = df_auto.dec.map(lambda x: float(str(x).split(".")[0] + "." + str(x).split(".")[1][:resolution]))

    # 列表匹配
    df_auto = pd.merge(poi_list, df_auto, how="inner")
    exist_list = os.listdir("/data/renhaoye/decals_2022/in_decals/jpg/")
    exist_poi = []
    for i in range(len(exist_list)):
        exist_poi.append([float(exist_list[i].split("_")[0]), float(exist_list[i].split("_")[1].split(".jpg")[0])])
    exist_poi = pd.DataFrame(exist_poi,  columns=["ra", "dec"])
    total = pd.concat([df_auto, exist_poi], ignore_index=True, verify_integrity=True, sort=True)
    # 删除合并后姓名里有重复的所有行
    total.drop_duplicates(subset=["ra", "dec"], keep=False, inplace=True)  # subset=['姓名']表示检查姓名这一列是否有重复内容，keep=False表示不保留重复内容
    urls = []
    for i in tqdm(range(len(total))):
        # for i in range(1):
        urls.append(get_url(total.iloc[i, 0], total.iloc[i, 1]))  # 0是ra，1是dec
    p = multiprocessing.Pool(processes=20)  # 进to程池
    _ = [p.apply_async(func=download_file, args=(i,)) for i in urls]
    p.close()
    p.join()
    # index = []
    # for i in range(len(urls)):
    #     index.append(i)
    # p = multiprocessing.Pool(25)
    # p.map(partial(download_file, filename=urls), index)
    # p.close()
    # p.join()