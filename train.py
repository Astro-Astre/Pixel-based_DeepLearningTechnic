# -*- coding: utf-8-*-

from torch import optim
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.focal_loss import *
from preprocess.data_handle import *
from models.SimpleCnn import *
from models.Xception import *
from models.DenseNet import *
from models.SwinTransformer import *
from torch.utils.data import DataLoader
from decals_dataset import *
from preprocess.data_handle import *
from args import *

#
# BATCH_SIZE = 64  # 批处理大小
# LEARNING_RATE = 0.0001  # 学习率
# CLASSES = 9
# EPOCHES = 150  # 训练次数
# MOMENTUM = 0.9
# SAVE_PATH = "/data/renhaoye/decals_2022/"  # the head of the directory to save

if data_config.rand_seed > 0:
    init_rand_seed(data_config.rand_seed)


class trainer:
    def __init__(self, loss_func, model, optimizer, config):
        self.loss_func = loss_func
        self.model = model
        self.optimizer = optimizer
        self.config = config

    def train(self, train_loader, valid_loader, test_loader):
        print("你又来炼丹辣！")
        losses = []
        acces = []
        start = -1
        mkdir(self.config.model_path + "log/")
        writer = torch.utils.tensorboard.SummaryWriter(self.config.model_path + "log/")
        for epoch in range(start + 1, self.config.epochs):
            train_loss = 0
            train_acc = 0
            self.model.train()
            for i, (X, label) in enumerate(train_loader):  # 遍历train_loader
                label = torch.as_tensor(label, dtype=torch.long)
                X, label = X.to(self.config.device), label.to(self.config.device)
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("model_cnn"):
                out = self.model(X)  # 正向传播
                loss_value = self.loss_func(out, label)  # 求损失值
                self.loss_func.zero_grad()  # 优化器梯度归零
                loss_value.backward()  # 反向转播，刷新梯度值
                self.loss_func.step()
                train_loss += float(loss_value)  # 计算损失
                _, pred = out.max(1)  # get the predict label
                num_correct = (pred == label).sum()  # compute the sum of correct pred
                acc = int(num_correct) / X.shape[0]  # compute the precision
                train_acc += acc  # accumilate the acc to compute the
                # print(prof.key_averages().table(sort_by="gpu_time_total", row_limit=10))
            writer.add_scalar('Training loss by steps', train_loss / len(train_loader), epoch)
            writer.add_scalar('Training accuracy by steps', train_acc / len(train_loader), epoch)
            writer.add_figure("Confusion matrix", cf_metrics(train_loader,
                                                             self.model, "/data/renhaoye/decals_2022/1.png",
                                                             False), epoch)
            losses.append(train_loss / len(train_loader))
            acces.append(100 * train_acc / len(train_loader))
            print("epoch: ", epoch)
            print("loss: ", train_loss / len(train_loader))
            print("accuracy:", train_acc / len(train_loader))

            eval_losses = []
            eval_acces = []
            eval_loss = 0
            eval_acc = 0
            with torch.no_grad():
                self.model.eval()
                for X, label in test_loader:
                    label = torch.as_tensor(label, dtype=torch.long)
                    X, label = X.to(self.config.device), label.to(self.config.device)
                    test_out = self.model(X)
                    test_loss = self.loss_func(test_out, label)
                    eval_loss += float(test_loss)
                    _, pred = test_out.max(1)
                    num_correct = (pred == label).sum()
                    acc = int(num_correct) / X.shape[0]
                    eval_acc += acc
            writer.add_figure("Confusion matrix valid",
                              cf_metrics(valid_loader, self.model, "/data/renhaoye/decals_2022/1.png", False),
                              epoch)
            eval_losses.append(eval_loss / len(test_loader))
            eval_acces.append(eval_acc / len(test_loader))
            writer.add_scalar('Testing loss by steps', eval_loss / len(test_loader), epoch)
            writer.add_scalar('Testing accuracy by steps', eval_acc / len(test_loader), epoch)
            print("test_loss: " + str(eval_loss / len(test_loader)))
            print("test_acc:" + str(eval_acc / len(test_loader)) + '\n')
            checkpoint = {
                "net": self.model.state_dict(),
                'optimizer': self.loss_func.state_dict(),
                "epoch": epoch
            }
            mkdir('%s/checkpoint' % self.config.model_path)
            torch.save(checkpoint, '%s/checkpoint/ckpt_best_%s.pth' % (self.config.model_path, str(epoch)))
            torch.save(self.model, '%s/model_%d.model' % (self.config.model_path, epoch))


train_data = DecalsDataset(annotations_file=data_config.train_file, transform=data_config.transfer)
train_loader = DataLoader(dataset=train_data, batch_size=data_config.batch_size,
                          shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)
test_data = DecalsDataset(annotations_file=data_config.valid_file, transform=data_config.transfer)
test_loader = DataLoader(dataset=test_data, batch_size=data_config.batch_size,
                         shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)
valid_data = DecalsDataset(annotations_file=data_config.test_file, transform=data_config.transfer)
valid_loader = DataLoader(dataset=valid_data, batch_size=data_config.batch_size,
                          shuffle=False, num_workers=data_config.WORKERS, pin_memory=True)

model = eval(data_config.model_name)(data_config.model_parm)
model.to(data_config.device)

loss_func = eval(data_config.loss_func)(data_config.loss_func_parm)
optimizer = eval(data_config.optimizer)(model.parameters(), **data_config.optimizer_parm)

Trainer = trainer(loss_func=loss_func, model=model, optimizer=optimizer, config=data_config)
Trainer.train(train_loader=train_loader, test_loader=test_loader, valid_loader=valid_loader)

#
# from torch import optim
# from torch import nn
# import torch
# from torch.utils.tensorboard import SummaryWriter
# from torch.nn import functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from models.focal_loss import *
# from preprocess.data_handle import *
# from models.SimpleCnn import *
# from models.Xception import *
# from models.DenseNet import *
# from models.SwinTransformer import *
#
# torch.manual_seed(1926)  # 设置随机数种子，确保结果可重复
# BATCH_SIZE = 64  # 批处理大小
# LEARNING_RATE = 0.0001  # 学习率
# CLASSES = 9
# EPOCHES = 150  # 训练次数
# MOMENTUM = 0.9
# SAVE_PATH = "/data/renhaoye/decals_2022/"  # the head of the directory to save
#
#
# # noinspection PyUnresolvedReferences
# def trainModel(model_pkg, flag, last_epoch, train_loader, test_loader, validation_loader):
#     """
#     训练
#     """
#     # model = DecalsModel()
#     # model = denseNet121Decals()
#     # model = denseNet169Decals()
#     # model = denseNet201Decals()
#     model = denseNet264Decals()
#     # model = Swin_T(10)
#     # model = x_ception(7)
#     device = "cuda:0"
#     # model = nn.DataParallel(model)
#     model.to(device)
#     mkdir(model_pkg)
#     # weight = [54528 / 23136, 54528 / 30435, 54528 / 17196, 54528 / 21237, 54528 / 22099, 54528 / 54528, 54528 / 12928]
#     # weight = [34101 / 23424, 34101 / 24737, 34101 / 26559, 34101 / 21502, 34101 / 12879, 34101 / 34101, 34101 / 13088]
#     # weight = [1 - (23424 / 156290), 1 - (24737 / 156290), 1 - (26559 / 156290), 1 - (21502 / 156290),
#     #           1 - (12879 / 156290), 1 - (34101 / 156290), 1 - (13088 / 156290)]
#     weight = [1 - (23424 / 174352), 1 - (24737 / 174352), 1 - (26559 / 174352), 1 - (21502 / 174352),
#               1 - (12813 / 174352), 1 - (9560 / 174352), 1 - (31245 / 174352), 1 - (11424 / 174352), 1 - (13088 / 174352)]
#     # criterion = nn.CrossEntropyLoss()  # loss function
#     criterion = focal_loss(alpha=weight, gamma=2, num_classes=9)
#     # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     losses = []  # a list of train_loss
#     acces = []  # a list of train_acc
#     start = -1
#     if flag:  # flag = True continue training
#         path_checkpoint = '%s/checkpoint/ckpt_best_%d.pth' % (model_pkg, last_epoch)  # 断点路径
#         checkpoint = torch.load(path_checkpoint)  # 加载断点
#         model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
#         optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
#         start = checkpoint['epoch']  # 读取上次的epoch
#         print('start epoch: ', last_epoch + 1)
#
#     print("Hello training!")
#     mkdir(model_pkg + "log/")
#     writer = torch.utils.tensorboard.SummaryWriter(model_pkg + "log/")
#     for epoch in range(start + 1, EPOCHES):
#         train_loss = 0
#         train_acc = 0
#         model.train()
#         for i, (X, label) in enumerate(train_loader):  # 遍历train_loader
#             label = torch.as_tensor(label, dtype=torch.long)
#             X, label = X.to(device), label.to(device)
#             # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#             #     with record_function("model_cnn"):
#             out = model(X)  # 正向传播
#             loss_value = criterion(out, label)  # 求损失值
#             optimizer.zero_grad()  # 优化器梯度归零
#             loss_value.backward()  # 反向转播，刷新梯度值
#             optimizer.step()
#             train_loss += float(loss_value)  # 计算损失
#             _, pred = out.max(1)  # get the predict label
#             num_correct = (pred == label).sum()  # compute the sum of correct pred
#             acc = int(num_correct) / X.shape[0]  # compute the precision
#             train_acc += acc  # accumilate the acc to compute the
#             # print(prof.key_averages().table(sort_by="gpu_time_total", row_limit=10))
#         writer.add_scalar('Training loss by steps', train_loss / len(train_loader), epoch)
#         writer.add_scalar('Training accuracy by steps', train_acc / len(train_loader), epoch)
#         writer.add_figure("Confusion matrix", cf_metrics(train_loader,
#                                                          model, "/data/renhaoye/decals_2022/1.png",
#                                                          False), epoch)
#         losses.append(train_loss / len(train_loader))
#         acces.append(100 * train_acc / len(train_loader))
#         print("epoch: ", epoch)
#         print("loss: ", train_loss / len(train_loader))
#         print("accuracy:", train_acc / len(train_loader))
#
#         eval_losses = []
#         eval_acces = []
#         eval_loss = 0
#         eval_acc = 0
#         with torch.no_grad():
#             model.eval()
#             for X, label in test_loader:
#                 label = torch.as_tensor(label, dtype=torch.long)
#                 X, label = X.to(device), label.to(device)
#                 test_out = model(X)
#                 test_loss = criterion(test_out, label)
#                 eval_loss += float(test_loss)
#                 _, pred = test_out.max(1)
#                 num_correct = (pred == label).sum()
#                 acc = int(num_correct) / X.shape[0]
#                 eval_acc += acc
#         writer.add_figure("Confusion matrix valid",
#                           cf_metrics(validation_loader, model, "/data/renhaoye/decals_2022/1.png", False),
#                           epoch)
#         eval_losses.append(eval_loss / len(test_loader))
#         eval_acces.append(eval_acc / len(test_loader))
#         writer.add_scalar('Testing loss by steps', eval_loss / len(test_loader), epoch)
#         writer.add_scalar('Testing accuracy by steps', eval_acc / len(test_loader), epoch)
#         print("test_loss: " + str(eval_loss / len(test_loader)))
#         print("test_acc:" + str(eval_acc / len(test_loader)) + '\n')
#         checkpoint = {
#             "net": model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             "epoch": epoch
#         }
#         mkdir('%s/checkpoint' % model_pkg)
#         torch.save(checkpoint, '%s/checkpoint/ckpt_best_%s.pth' % (model_pkg, str(epoch)))
#         torch.save(model, '%s/model_%d.model' % (model_pkg, epoch))
