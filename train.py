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
from grad_cam_utils import *
from models.data_parallel import *
import pickle as pkl

if data_config.rand_seed > 0:
    init_rand_seed(data_config.rand_seed)

torch.nn.CrossEntropyLoss
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
        mkdir(self.config.model_path)
        mkdir(self.config.model_path + "log/")
        mkdir(self.config.model_path + "cfm/")
        writer = torch.utils.tensorboard.SummaryWriter(self.config.model_path + "log/")
        writer.add_graph(model.module, torch.rand(1, 3, 256, 256).cuda())
        info = data_config()
        if data_config.resume:  # contin = True continue training
            path_checkpoint = '%s/checkpoint/ckpt_best_%d.pth' % (data_config.model_path, data_config.last_epoch)  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点
            model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start = checkpoint['epoch']  # 读取上次的epoch
            print(start)
            print('start epoch: ', data_config.last_epoch + 1)
        with open(data_config.model_path + "info.txt", "w") as w:
            for each in info.__dir__():
                attr_name = each
                attr_value = info.__getattribute__(each)
                w.write(str(attr_name) + ':' + str(attr_value) + "\n")
        for epoch in range(start + 1, self.config.epochs):
            train_loss = 0
            train_acc = 0
            self.model.train()
            for i, (X, label) in enumerate(train_loader):  # 遍历train_loader
                label = torch.as_tensor(label, dtype=torch.long)
                '''******** - 非分布式 -********'''
                # X, label = X.to(self.config.device), label.to(self.config.device)
                '''******** - 分布式 -********'''
                X = X.cuda()
                label = label.cuda()
                # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #     with record_function("model_cnn"):
                out = self.model(X)  # 正向传播
                loss_value = self.loss_func(out, label)  # 求损失值
                self.optimizer.zero_grad()  # 优化器梯度归零
                loss_value.backward()  # 反向转播，刷新梯度值
                self.optimizer.step()
                train_loss += float(loss_value)  # 计算损失
                _, pred = out.max(1)  # get the predict label
                num_correct = (pred == label).sum()  # compute the sum of correct pred
                acc = int(num_correct) / X.shape[0]  # compute the precision
                train_acc += acc  # accumilate the acc to compute the
                # print(prof.key_averages().table(sort_by="gpu_time_total", row_limit=10))
            writer.add_scalar('Training loss by steps', train_loss / len(train_loader), epoch)
            writer.add_scalar('Training accuracy by steps', train_acc / len(train_loader), epoch)

            # writer.add_figure("Confusion matrix", cf_metrics(train_loader, self.model, False), epoch)
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
                    # X, label = X.to(self.config.device), label.to(self.config.device)
                    X = X.cuda()
                    label = label.cuda()
                    test_out = self.model(X)
                    test_loss = self.loss_func(test_out, label)
                    eval_loss += float(test_loss)
                    _, pred = test_out.max(1)
                    num_correct = (pred == label).sum()
                    acc = int(num_correct) / X.shape[0]
                    eval_acc += acc
            cfm = cf_m(valid_loader, self.model)
            with open(self.config.model_path + "cfm/epoch_%d_test.dat" % epoch, "wb") as w:
                pkl.dump(cfm, w)
            writer.add_figure("Confusion matrix test",
                              cf_map(cfm),
                              epoch)
            # target_layers = [model.module.exit_flow.conv]
            # for batch_idx, (features, targets) in enumerate(train_loader):
            #     if batch_idx == 0:
            #         input = torch.unsqueeze(features[0], dim=0)
            #         writer.add_image('raw', input[0], epoch, dataformats="CHW")
            #         cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            #         for target in range(data_config.num_class):
            #             grayscale_cam = cam(input_tensor=input, target_category=target)
            #             grayscale_cam = grayscale_cam[0, :]
            #             img = np.zeros((3, 256, 256))
            #             visualization = show_cam_on_image(img.astype(dtype=np.float32),
            #                                               grayscale_cam,
            #                                               use_rgb=True)
            #             writer.add_image('%d grad-cam' % target, visualization, epoch, dataformats="HWC")
            eval_losses.append(eval_loss / len(test_loader))
            eval_acces.append(eval_acc / len(test_loader))
            writer.add_scalar('Testing loss by steps', eval_loss / len(test_loader), epoch)
            writer.add_scalar('Testing accuracy by steps', eval_acc / len(test_loader), epoch)
            print("test_loss: " + str(eval_loss / len(test_loader)))
            print("test_acc:" + str(eval_acc / len(test_loader)) + '\n')
            checkpoint = {
                "net": self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "epoch": epoch
            }
            mkdir('%s/checkpoint' % self.config.model_path)
            torch.save(checkpoint, '%s/checkpoint/ckpt_best_%s.pth' % (self.config.model_path, str(epoch)))
            torch.save(self.model.module, '%s/model_%d.pt' % (self.config.model_path, epoch))


train_data = DecalsDataset(annotations_file=data_config.train_file, transform=data_config.transfer)
train_loader = DataLoader(dataset=train_data, batch_size=data_config.batch_size,
                          shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)

test_data = DecalsDataset(annotations_file=data_config.valid_file, transform=data_config.transfer)
test_loader = DataLoader(dataset=test_data, batch_size=data_config.batch_size,
                         shuffle=True, num_workers=data_config.WORKERS, pin_memory=True)

valid_data = DecalsDataset(annotations_file=data_config.test_file, transform=data_config.transfer)
valid_loader = DataLoader(dataset=valid_data, batch_size=data_config.batch_size,
                          shuffle=False, num_workers=data_config.WORKERS, pin_memory=True)

model = eval(data_config.model_name)(**data_config.model_parm)
model = model.cuda()
device_ids = [0, 1]
# model = torch.nn.DataParallel(model, device_ids=device_ids)
model = BalancedDataParallel(12, model, dim=0, device_ids=[0, 1])

loss_func = eval(data_config.loss_func)(**data_config.loss_func_parm)
optimizer = eval(data_config.optimizer)(model.parameters(), **data_config.optimizer_parm)

Trainer = trainer(loss_func=loss_func, model=model, optimizer=optimizer, config=data_config)
Trainer.train(train_loader=train_loader, test_loader=test_loader, valid_loader=valid_loader)