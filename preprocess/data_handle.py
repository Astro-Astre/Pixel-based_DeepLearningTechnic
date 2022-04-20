import datetime
import os
import pickle
import numpy as np
import torch
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from astropy.io import fits
import torchvision.transforms as transforms
import random
from args import *


def get_weight(num_samples: list):
    sum = 0
    for i in num_samples:
        sum += i
    for i in range(len(num_samples)):
        num_samples[i] = 1 - num_samples[i] / sum
    return num_samples


def changeAxis(data: np.ndarray):
    """
    input: ndarray, C×H×W, should do before ToTensor()
    """
    img = np.swapaxes(data, 0, 2)
    img = np.swapaxes(img, 0, 1)
    return img


def init_rand_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    # torch.backends.cudnn.deterministic = True # 这个可能会导致训练过程非常慢


def normalization(img: np.ndarray, shape: str, lock: bool):
    """
    :param img: CHW和HWC均可
    :param shape: 传入CHW或者HWC，输出同一形式
    :param lock: True时，三通道一起归一化，否则分通道归一化
    :return: 返回CHW或HWC
    """

    def compute(data: np.ndarray):
        h, w = data.shape
        data = data.reshape(-1)
        max = np.max(data)
        min = np.min(data)
        if max == min:
            return 65535
        else:
            norm_data = (data - min) / (max - min)
            return norm_data.reshape((h, w))

    if lock:
        original_shape = img.shape
        flatten = img.reshape(-1)
        normalized = (flatten - flatten.min()) / (flatten.max() - flatten.min())
        return normalized.reshape(original_shape)
    else:
        if shape == "CHW" or shape == "chw":
            g, r, z = img[0], img[1], img[2]
            norm_g, norm_r, norm_z = compute(g), compute(r), compute(z)
            if type(norm_g) == int or type(norm_r) == int or type(norm_z) == int:
                return 65535
            else:
                return np.array((norm_g, norm_r, norm_z))
        elif shape == "HWC" or shape == "hwc":
            g, r, z = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            norm_g, norm_r, norm_z = compute(g), compute(r), compute(z)
            if type(norm_g) == int or type(norm_r) == int or type(norm_z) == int:
                return 65535
            else:
                return np.concatenate(
                    (norm_g.reshape(256, 256, 1), norm_r.reshape(256, 256, 1), norm_z.reshape(256, 256, 1)), axis=2)
        else:
            raise RuntimeError


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
        hdulist.writeto(filename)
        hdulist.close()
    elif data.shape[-1] == 3:
        g, r, z = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        data = np.array((g, r, z))
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename)
        hdulist.close()
    elif data.shape[0] == 3:
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename)
        hdulist.close()
    else:
        raise RuntimeError


# def color_balance(img: np.ndarray, shape: str, region: str, kernel_size: list = [60, 60]):
#     """
#     三通道直方图平衡
#     :param img: 输入图像，CHW和HWC均可，需要在shape中输入
#     :param kernel_size: 核大小，长宽不必相同
#     :param shape: 应当为CHW或HWC
#     :param region: 全局模式则输入global，否则输入kernel使用中心核模式，无默认大小
#     :return: 根据输入shape返回对应形式的ndarray
#     """
#     if region == "global":
#         if shape == "CHW" or shape == "chw":
#             r, g, b = img[0], img[1], img[2]
#             R, G, B = np.mean(r), np.mean(g), np.mean(b)
#             K = (R + G + B) / 3
#             Kr, Kg, Kb = K / R, K / G, K / B
#             return np.array((Kr * r, Kg * g, Kb * b))
#         elif shape == "HWC" or shape == "hwc":
#             r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
#             R, G, B = np.mean(r), np.mean(g), np.mean(b)
#             K = (R + G + B) / 3
#             Kr, Kg, Kb = K / R, K / G, K / B
#             return np.concatenate(
#                 (Kr * r.reshape(256, 256, 1), Kg * g.reshape(256, 256, 1), Kb * b.reshape(256, 256, 1)), axis=2)
#         else:
#             raise RuntimeError
#     elif region == "kernel":
#         h_k, w_k = kernel_size[0], kernel_size[1]  # 核的长宽
#         h, w = img[0].shape  # 图像长（纵向）宽（横向）
#         h_l, h_r = int(h / 2) - h_k, int(h / 2) + h_k
#         w_l, w_r = int(w / 2) - w_k, int(w / 2) + w_k
#         r, g, b = img[0], img[1], img[2]
#         r_kernel, g_kernel, b_kernel = img[0, h_l:h_r, w_l: w_r], img[1, h_l:h_r, w_l: w_r], img[2, h_l:h_r, w_l: w_r]
#         R, G, B = np.mean(r_kernel), np.mean(g_kernel), np.mean(b_kernel)
#         K = (R + G + B) / 3
#         Kr, Kg, Kb = K / R, K / G, K / B
#         if shape == "CHW" or shape == "chw":
#             return np.array((Kr * r, Kg * g, Kb * b))
#         elif shape == "HWC" or shape == "hwc":
#             return np.concatenate(
#                 (Kr * r.reshape(256, 256, 1), Kg * g.reshape(256, 256, 1), Kb * b.reshape(256, 256, 1)), axis=2)
#         else:
#             raise RuntimeError
#     else:
#         raise RuntimeError

#
# def median_balance(img: np.ndarray, shape: str, region: str, kernel_size: list = [60, 60]):
#     """
#     三通道直方图平衡
#     :param img: 输入图像，CHW和HWC均可，需要在shape中输入
#     :param kernel_size: 核大小，长宽不必相同
#     :param shape: 应当为CHW或HWC
#     :param region: 全局模式则输入global，否则输入kernel使用中心核模式，无默认大小
#     :return: 根据输入shape返回对应形式的ndarray
#     """
#     if region == "global":
#         if shape == "CHW" or shape == "chw":
#             r, g, b = img[0], img[1], img[2]
#             R, G, B = np.median(r), np.median(g), np.median(b)
#             K = (R + G + B) / 3
#             Kr, Kg, Kb = K / R, K / G, K / B
#             return np.array((Kr * r, Kg * g, Kb * b))
#         elif shape == "HWC" or shape == "hwc":
#             r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
#             R, G, B = np.median(r), np.median(g), np.median(b)
#             K = (R + G + B) / 3
#             Kr, Kg, Kb = K / R, K / G, K / B
#             return np.concatenate(
#                 (Kr * r.reshape(256, 256, 1), Kg * g.reshape(256, 256, 1), Kb * b.reshape(256, 256, 1)), axis=2)
#         else:
#             raise RuntimeError
#     elif region == "kernel":
#         h_k, w_k = kernel_size[0], kernel_size[1]  # 核的长宽
#         h, w = img[0].shape  # 图像长（纵向）宽（横向）
#         h_l, h_r = int(h / 2) - h_k, int(h / 2) + h_k
#         w_l, w_r = int(w / 2) - w_k, int(w / 2) + w_k
#         r, g, b = img[0], img[1], img[2]
#         r_kernel, g_kernel, b_kernel = img[0, h_l:h_r, w_l: w_r], img[1, h_l:h_r, w_l: w_r], img[2, h_l:h_r, w_l: w_r]
#         R, G, B = np.median(r_kernel), np.median(g_kernel), np.median(b_kernel)
#         K = (R + G + B) / 3
#         Kr, Kg, Kb = K / R, K / G, K / B
#         if shape == "CHW" or shape == "chw":
#             return np.array((Kr * r, Kg * g, Kb * b))
#         elif shape == "HWC" or shape == "hwc":
#             return np.concatenate(
#                 (Kr * r.reshape(256, 256, 1), Kg * g.reshape(256, 256, 1), Kb * b.reshape(256, 256, 1)), axis=2)
#         else:
#             raise RuntimeError
#     else:
#         raise RuntimeError


def mtf(data: np.ndarray, m: float):
    """
    非线性变换
    :param data: 输入数据
    :param m: midtones值
    :return: 结果
    """
    return ((m - 1) * data) / ((2 * m - 1) * data - m)


# def rgb2gray(img: np.ndarray, shape: str):
#     """
#     rgb转灰度图
#     :param img: 图像
#     :param channel_factor: 三通道权重因子，应为list
#     :return:
#     """
#     return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def channel_cut(gray):
    highlight = 1.
    hist, bar = np.histogram(gray.reshape(-1), bins=65536)
    cdf = hist.cumsum()
    shadow_index = np.argwhere(cdf > 0.001 * gray.reshape(-1).shape[0])[0]
    shadow = bar[shadow_index]
    midtones = np.median(gray) - shadow
    gray[gray < shadow] = shadow
    gray[gray > highlight] = 1.
    gray = gray.reshape(-1)
    norm_data = (gray - gray.min()) / (gray.max() - gray.min())
    gray = norm_data.reshape((256, 256))
    median = np.median(mtf(gray, midtones))
    set_median = 1 / 8
    weight = median / set_median
    right_midtones = weight * midtones
    return mtf(gray, right_midtones)


def auto_scale(data: np.ndarray):
    g, r, z = channel_cut(data[0]), channel_cut(data[1]), channel_cut(data[2])
    img = np.array((g, r, z))
    img[img < 0] = 0
    img[img > 1] = 1.
    return img


def chw2hwc(img):
    ch1, ch2, ch3 = img[0], img[1], img[2]
    h, w = ch1.shape
    return np.concatenate((ch1.reshape(h, w, 1), ch2.reshape(h, w, 1), ch3.reshape(h, w, 1)), axis=2)


def hwc2chw(img):
    ch1, ch2, ch3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return np.array((ch1, ch2, ch3))


# def normalization(img):
#     """
#     0-1 normalization
#     img: img of 3 channels
#     :return
#     """
#     global g_signal, r_signal, z_signal
#     data_max = np.array([np.max(img[0]), np.max(img[1]), np.max(img[2])])
#     data_min = np.array([np.min(img[0]), np.min(img[1]), np.min(img[2])])
#
#     def compute(data, max, min):
#         if max == min:
#             return 0, True
#         else:
#             return (data - min) / (max - min), False
#
#     for j in range(256):
#         img[0][j], g_signal = compute(img[0][j], data_max[0], data_min[0])
#         img[1][j], r_signal = compute(img[1][j], data_max[1], data_min[1])
#         img[2][j], z_signal = compute(img[2][j], data_max[2], data_min[2])
#     if g_signal or r_signal or z_signal:
#         return img, True
#     elif g_signal == r_signal == z_signal == False:
#         return img, False


def load_img(file, transform):
    """
    加载图像，dat和fits均支持，不过仅支持CxHxW
    :param filename: 传入文件名，应当为CHW
    :return: 返回CHW的ndarray
    """
    if ".fits" in file:
        with fits.open(file) as hdul:
            return hdul[0].data.astype(np.float32)
    elif ".dat" in file:
        with open(file, "rb") as f:
            return pickle.load(f)
    else:
        raise TypeError


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def mk_model_dir(info):
    """
    返回文件夹名：model_info_datetime
    """
    model_package = 'model_%s' % info
    if not model_package[-1] == '_':
        model_package += '_'
    model_package += str(datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    return model_package


def save_fits(data: np.ndarray, filename: str):
    """
    将ndarray保存成fits文件
    :param data: 待保存数据
    :param filename: 保存文件名
    :return:
    """
    if len(data.shape) == 2:
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename)
        hdulist.close()
    elif data.shape[-1] == 3:
        g, r, z = data[:, :, 0], data[:, :, 1], data[:, :, 2]
        data = np.array((g, r, z))
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename)
        hdulist.close()
    elif data.shape[0] == 3:
        hdu = fits.PrimaryHDU(data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename)
        hdulist.close()
    else:
        raise RuntimeError


def save_dat(object, dir):
    with open(dir, 'wb') as f:
        if type(object) is None:
            print("error")
        else:
            pickle.dump(object, f)


def cf_metrics(loader, model, save: bool = False):
    y_pred = []  # save predction
    y_true = []  # save ground truth
    device = "cuda:0"
    # iterate over data
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)  # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth
    classes = data_config.classes
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis],
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 10))
    if save:
        sns.heatmap(df_cm, annot=True).get_figure()
        plt.savefig(data_config.metrix_save_path)
    else:
        return sns.heatmap(df_cm, annot=True).get_figure()
