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


def changeAxis(data: np.ndarray):
    """
    input: ndarray, C×H×W, should do before ToTensor()
    """
    img = np.swapaxes(data, 0, 2)
    img = np.swapaxes(img, 0, 1)
    return img


def normalization(img):
    """
    0-1 normalization
    img: img of 3 channels
    :return
    """
    global g_signal, r_signal, z_signal
    data_max = np.array([np.max(img[0]), np.max(img[1]), np.max(img[2])])
    data_min = np.array([np.min(img[0]), np.min(img[1]), np.min(img[2])])

    def compute(data, max, min):
        if max == min:
            return 0, True
        else:
            return (data - min) / (max - min), False
    for j in range(256):
        img[0][j], g_signal = compute(img[0][j], data_max[0], data_min[0])
        img[1][j], r_signal = compute(img[1][j], data_max[1], data_min[1])
        img[2][j], z_signal = compute(img[2][j], data_max[2], data_min[2])
    if g_signal or r_signal or z_signal:
        return img, True
    elif g_signal == r_signal == z_signal == False:
        return img, False


def load_dat(file, transform):
    with open(file, "rb") as f:
        img = pickle.load(f)
        img = changeAxis(img)
        img = transform(img)
        return img


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


def save_dat(object, dir):
    with open(dir, 'wb') as f:
        if type(object) is None:
            print("error")
        else:
            pickle.dump(object, f)
    # print(type(object))
    # print(object.shape)
    # hdu = fits.PrimaryHDU(object)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(dir.split(".dat")[0] + ".fits")


def cf_metrics(loader, model, save_path, save: bool = False):
    y_pred = []  # save predction
    y_true = []  # save ground truth
    # iterate over data
    for inputs, labels in loader:
        output = model(inputs)  # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth
    classes = ("merger", "smoothRounded", "smoothInBetween", "smoothCigarShaped", "edgeOn", "diskNoWeakBar", "diskStrongBar")
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis],
                         index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 10))
    if save:
        sns.heatmap(df_cm, annot=True).get_figure()
        plt.savefig(save_path)
    else:
        return sns.heatmap(df_cm, annot=True).get_figure()
