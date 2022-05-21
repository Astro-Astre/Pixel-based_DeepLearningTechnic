from args import *
from torch import optim
from torch import nn
import torch
from metrix import *
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from models.focal_loss import *
from models.Xception import *
from models.SwinTransformer import *
from torch.utils.data import DataLoader
from decals_dataset import *
from preprocess.data_handle import *
from functools import partial
import multiprocessing

MODEL_PATH = "/data/renhaoye/decals_2022/trained_model/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-BEST_2/model_6.pt"
DATA_PATH = "/data/renhaoye/decals_2022/out_decals/scaled/"


def pred(i, rows, w):
    data = load_img(DATA_PATH + rows[i], transform=None)
    x = torch.from_numpy(data)
    y = model(x.to("cuda:0").unsqueeze(0))
    pred = (torch.max(torch.exp(y), 1)[1]).data.cpu().numpy()
    w.writelines(str(rows[i]) + " " + str(pred)[1] + "\n")


if __name__ == '__main__':
    if data_config.rand_seed > 0:
        init_rand_seed(data_config.rand_seed)
    out_decals = os.listdir(DATA_PATH)
    # torch.cuda.set_device("cuda:0")
    model = torch.load(MODEL_PATH)
    device_ids = [0, 1]
    model.to("cuda:0")
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    with open("/data/renhaoye/decals_2022/test.txt", "w+") as w:
        w.writelines("loc label\n")
    w = open("/data/renhaoye/decals_2022/test.txt", "a")
    # for i in range(len(out_decals)):
    for i in range(10):
        pred(i, out_decals, w)
    w.close()
    # index = []
    # for i in range(len(out_decals)):
    #     index.append(i)
    # p = multiprocessing.Pool(8)
    # p.map(partial(augmentation, rows=out_decals, w=w), index)
    # p.close()
    # p.join()
