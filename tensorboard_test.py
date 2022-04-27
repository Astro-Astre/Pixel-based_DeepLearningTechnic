# -*- coding: utf-8-*-

from args import *
from torch import optim
from torch import nn
import torch
from metrix import *
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.focal_loss import *
from preprocess.data_handle import *
from models.SimpleCnn import *
from models.Xception import *
from models.efficientnet import *
from models.DenseNet import *
from models.SwinTransformer import *
from torch.utils.data import DataLoader
from decals_dataset import *
from preprocess.data_handle import *
from models.dense import *


if data_config.rand_seed > 0:
    init_rand_seed(data_config.rand_seed)


class trainer:
    def __init__(self, loss_func, model, optimizer, config):
        self.loss_func = loss_func
        self.model = model
        self.optimizer = optimizer
        self.config = config

    def train(self):
        writer = torch.utils.tensorboard.SummaryWriter("/data/renhaoye/decals_2022/runs/")
        # init_img = torch.ones((1, 3, 256, 256))
        # writer.add_graph(self.model, init_img)
        writer.add_graph(model.module, torch.rand(1, 3, 256, 256).cuda())
        writer.close()


model = eval(data_config.model_name)(**data_config.model_parm)

model = model.cuda()
device_ids = [0, 1]
model = torch.nn.DataParallel(model, device_ids=device_ids)
loss_func = eval(data_config.loss_func)(**data_config.loss_func_parm)
optimizer = eval(data_config.optimizer)(model.parameters(), **data_config.optimizer_parm)


Trainer = trainer(loss_func=loss_func, model=model, optimizer=optimizer, config=data_config)
Trainer.train()
