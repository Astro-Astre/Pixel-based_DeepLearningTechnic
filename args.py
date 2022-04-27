import os
import torchvision.transforms as transforms
from preprocess.data_handle import *


class data_config:
    input_channel = 3
    num_class = 7
    resume = False
    last_epoch = 5
    # model_name = "efficientnet_b5"
    # model_name = "denseNet201"
    model_name = "x_ception"
    # model_parm = {'input_channels': input_channel, 'num_class': num_class}
    model_parm = {}
    '''***********- dataset and directory -*************'''
    root_path = '/data/renhaoye/decals_2022/'
    train_file = '/data/renhaoye/decals_2022/train-CLASS_7_2-STF.txt'
    valid_file = '/data/renhaoye/decals_2022/valid-CLASS_7_2-STF.txt'
    test_file = '/data/renhaoye/decals_2022/test-CLASS_7_2-STF.txt'
    transfer = transforms.Compose([transforms.ToTensor()])

    '''***********- Hyper Arguments -*************'''
    WORKERS = 20
    epochs = 100
    batch_size = 32
    gamma = 2
    rand_seed = 1926
    lr = 0.0001
    momentum = 0.9
    # weight = [54528 / 23136, 54528 / 30435, 54528 / 17196, 54528 / 21237, 54528 / 22099, 54528 / 54528, 54528 / 12928]
    # weight = [34101 / 23424, 34101 / 24737, 34101 / 26559, 34101 / 21502, 34101 / 12879, 34101 / 34101, 34101 / 13088]
    # weight = [1 - (23424 / 156290), 1 - (24737 / 156290), 1 - (26559 / 156290), 1 - (21502 / 156290),
    #           1 - (12879 / 156290), 1 - (34101 / 156290), 1 - (13088 / 156290)]
    # weight = [1 - (23424 / 174352), 1 - (24737 / 174352), 1 - (26559 / 174352), 1 - (21502 / 174352),
    #           1 - (12813 / 174352), 1 - (9560 / 174352), 1 - (31245 / 174352), 1 - (11424 / 174352),
    #           1 - (13088 / 174352)]
    # weight = get_weight([23424, 24737, 26559, 21502, 12813, 9650, 31425, 11424, 13088])  # 9
    # weight = get_weight([23424, 24737, 26559, 21502, 51516, 31245, 11424, 13088])  # 8
    weight = get_weight([13536, 24737, 26559, 17685, 11540, 14813, 6720])  # 7,threshold2
    # weight = get_weight([23424, 24737, 26559, 21502, 11540, 14813, 13088])  # 7,threshold_c
    # optimizer = "torch.optim.Adam"
    optimizer = "torch.optim.AdamW"
    optimizer_parm = {'lr': lr, 'weight_decay': 0.001}
    # loss_func = 'torch.nn.CrossEntropyLoss'
    # loss_func_parm = {}
    loss_func = 'focal_loss'
    loss_func_parm = {'alpha': weight, 'gamma': gamma, 'num_classes': num_class}
    device = "cuda:0"
    # local_rank = 0, 1
    multi_gpu = False
    other = "graduation"
    model_path = root_path + 'trained_model/%s-LR_%s-LOSS_%s-CLASS_%s-BATCHSIZE_%s-OPTIM_%s-OTHER_%s/' \
                 % (model_name, str(lr), loss_func, str(num_class), str(batch_size), optimizer, other)
    metrix_save_path = ""
    # classes = (
    #     "merger", "smoothRounded", "smoothInBetween", "smoothCigarShaped",
    #     "edgeOn", "diskNoWeakBar", "diskStrongBar")
    # classes = (
    #     "merger", "smoothRounded", "smoothInBetween", "smoothCigarShaped",
    #     "edgeOnBulge", "edgeOnNoBulge", "diskNoBar", "diskWeakBar", "diskStrongBar")    # 9
    classes = (
        "merger", "Rounded", "InBetween", "CigarShaped",
        "edgeOn", "diskNoBar", "diskStrongBar")  # 9
