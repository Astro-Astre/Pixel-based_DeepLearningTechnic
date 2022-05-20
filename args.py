import os
from preprocess.data_handle import *
import torchvision.transforms as transforms

'''
训练参数设置
'''


class data_config:
    input_channel = 3  # 输入通道数量，如果需要加mask，要改这里
    num_class = 7  # 类别储量，还要对应改weight，classes
    resume = False  # 断点续训
    last_epoch = 6  # 从哪个epoch开始训练
    # model_name = "x_ception"  # 模型方法名，eval()调用，
    model_name = "swin"  # 模型方法名，eval()调用，
    # model_parm = {'input_channels': input_channel, 'num_class': num_class}
    model_parm = {}  # 模型参数
    '''***********- dataset and directory -*************'''
    root_path = '/data/renhaoye/decals_2022/'  # 根目录
    origin = "decals"
    name = "BEST"
    train_file = '/data/renhaoye/decals_2022/dataset_txt/%s-CLASS_7-%s-train.txt' % (origin, name)  # 训练集txt文件
    valid_file = '/data/renhaoye/decals_2022/dataset_txt/%s-CLASS_7-%s-valid.txt' % (origin, name)  # 验证集txt文件
    test_file = '/data/renhaoye/decals_2022/dataset_txt/%s-CLASS_7-%s-test.txt' % (origin, name)  # 测试集txt文件
    transfer = transforms.Compose([transforms.ToTensor()])

    '''***********- Hyper Arguments -*************'''
    WORKERS = 12  # dataloader进程数量
    epochs = 30  # 训练总epoch
    batch_size = 32  # 批处理大小
    gamma = 2  # focal_loss超参数
    rand_seed = 1926  # 随机种子
    lr = 0.0001  # 学习率
    momentum = 0.9  # SGD的动量设置
    # weight = get_weight([13536, 24737, 26559, 17685, 11540, 14813, 6720])  # best
    weight = get_weight([13536 / 4, 24737, 26559, 17685, 11540, 14813, 6720 / 4])  # baseline
    # weight = get_weight([23424, 24737, 26559, 21502, 11540, 14813, 13088])  # 7,threshold_c
    # weight = get_weight([4*2283, 16972, 18245, 11842, 7529, 9991, 4*1122])  # sdss_weight
    # optimizer = "torch.optim.Adam"
    optimizer = "torch.optim.AdamW"  # 优化器方法名称，eval()调用
    optimizer_parm = {'lr': lr, 'weight_decay': 0.01}  # 优化器参数
    # loss_func = 'torch.nn.CrossEntropyLoss'  # 损失函数方法名称，eval()调用
    # loss_func_parm = {}  # 损失函数参数
    loss_func = 'focal_loss'
    loss_func_parm = {'alpha': weight, 'gamma': gamma, 'num_classes': num_class}
    device = "cuda:0"  # gpu
    # local_rank = 0, 1
    multi_gpu = False  # 多卡设置
    other = "swinT"  # 模型保存文件夹备注
    model_path = root_path + 'trained_model/%s-LR_%s-LS_%s-CLS_%s-BSZ_%s-OPT_%s-%s/' \
                 % (model_name, str(lr), loss_func, str(num_class), str(batch_size), optimizer.split(".")[-1], other)
    metrix_save_path = "/data/renhaoye/decals_2022/sdss_pred_newFL.jpg"
    classes = (
        "merger", "round", "between", "cigar",
        "edgeOn", "noBar", "strongBar")  # 类别名称
