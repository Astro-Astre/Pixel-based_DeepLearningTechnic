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
# MODEL_PATH = "/data/renhaoye/decals_2022/trained_model/x_ception-LR_0.0001-LOSS_focal_loss-CLASS_7-BATCHSIZE_32-OPTIM_torch.optim.AdamW-OTHER_graduation_weightdecay0.01/model_5.pt"
# MODEL_PATH = "/data/renhaoye/decals_2022/trained_model/x_ception-LR_0.0001-LOSS_torch.nn.CrossEntropyLoss-CLASS_7-BATCHSIZE_24-OPTIM_torch.optim.AdamW-OTHER_sdss_best/model_3.pt"
# MODEL_PATH = "/data/renhaoye/decals_2022/trained_model/x_ception-LR_0.0001-LOSS_focal_loss-CLASS_7-BATCHSIZE_32-OPTIM_torch.optim.AdamW-OTHER_graduation_weightdecay0.01/model_15.pt"
# MODEL_PATH = "/data/renhaoye/decals_2022/trained_model/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-BEST/model_15.pt"
# MODEL_PATH = "/data/renhaoye/decals_2022/trained_model/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-BEST_2/model_6.pt"
MODEL_PATH = "/data/renhaoye/decals_2022/trained_model/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-sdss/model_8.pt"


if __name__ == '__main__':
    if data_config.rand_seed > 0:
        init_rand_seed(data_config.rand_seed)
    # sdss_dir = "/data/renhaoye/sdss_dr7_decals_overlap/scaled"
    # decals_csv = "/data/renhaoye/decals_2022/fits.csv"
    # sdss_files = os.listdir(sdss_dir)
    # decals_df = pd.read_csv(decals_csv)
    # print(len(sdss_files), decals_df.info())
    # save_path = "/data/renhaoye/decals_2022/dataset_txt/sdss-CLASS_7-BEST-test.txt"
    save_path = "/data/renhaoye/decals_2022/dataset_txt/decals-CLASS_7-BEST-test.txt"
    sdss_data = DecalsDataset(annotations_file=save_path, transform=data_config.transfer)
    sdss_loader = DataLoader(dataset=sdss_data, batch_size=5,
                              shuffle=False, num_workers=6, pin_memory=True)
    model = torch.load(MODEL_PATH)
    device_ids = [0, 1]
    # model.to("cuda:0")
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    cfm = cf_m(sdss_loader, model)
    pic = cf_map(cfm)
    plt.savefig("/data/renhaoye/decals_2022/pred_decals_0519.png")

