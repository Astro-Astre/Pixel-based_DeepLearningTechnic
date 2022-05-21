from args import *
from torch import optim
from torch import nn
import torch
from metrix import *
from torch.nn import functional as F
from models.focal_loss import *
from models.Xception import *
from decals_dataset import *
from preprocess.data_handle import *

MODEL_PATH = "/data/renhaoye/decals_2022/trained_model/x_ception-LR_0.0001-LS_focal_loss-CLS_7-BSZ_32-OPT_AdamW-BEST_2/model_6.pt"
ROOT = "/data/renhaoye/decals_2022/"


def pred(path):
    data = load_img(path, transform=None)
    x = torch.from_numpy(data)
    y = model(x.to("cuda:0").unsqueeze(0))
    pred = F.softmax(torch.Tensor(y.cpu()), dim=1).detach().numpy()[0]
    prob = pred[np.argmax(pred)]
    return prob, int(np.argmax(pred))


if __name__ == '__main__':
    if data_config.rand_seed > 0:
        init_rand_seed(data_config.rand_seed)
    # torch.cuda.set_device("cuda:0")
    model = torch.load(MODEL_PATH)
    device_ids = [0, 1]
    model.to("cuda:0")
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    sdss = pd.read_csv(ROOT + "dataset_sdss_match_decals.csv", index_col=0)
    sdss_test = sdss.query("func == 'test'")
    decals = pd.read_csv(ROOT + "dataset_decals.csv", index_col=0)
    decals_test = decals.query("func == 'test'")
    decals_test['prob'] = -1.
    sdss_test['prob'] = -1.
    decals_test['pred'] = -1
    sdss_test['pred'] = -1
    for i in range(len(decals_test)):
    # for i in range(1):
        path = ROOT + "in_decals/decals_best/" + str(decals_test.iloc[i, 0]) + "_" + str(decals_test.iloc[i, 1]) + ".fits"
        decals_test.iloc[i, -2], decals_test.iloc[i, -1] = pred(path)
    for i in range(len(sdss_test)):
    # for i in range(1):
        path = ROOT + "in_decals/sdss_agmtn_scaled/" + sdss_test.iloc[i, 13].split("/")[-1]
        sdss_test.iloc[i, -2], sdss_test.iloc[i, -1] = pred(path)
    decals_test.to_csv(ROOT + "decals_test_prob.csv")
    sdss_test.to_csv(ROOT + "sdss_test_prob.csv")
