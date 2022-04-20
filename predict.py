from preprocess.data_handle import *
from train import *
from decals_dataset import *
from models.Xception import x_ception
import pandas as pd
import time


def predict(model_name, dir):
    transfer = transforms.Compose([
        transforms.ToTensor(),
    ])
    model = x_ception(7)
    model = nn.DataParallel(model)
    device = "cuda:0"
    model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage).state_dict())
    model.to(device)
    torch.no_grad()
    img_dir = os.listdir(dir)
    with open("/data/renhaoye/decals_2022/out_decals_pred_9_015.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["loc", "label", "time_cost"])
        for file in img_dir:
            line = [dir + file]
            img = load_img(dir + file, transfer)
            start = time.time()
            out = model(img.to(device).unsqueeze(0))
            _, pred = out.max(1)
            line.append(pred.item())
            line.append(time.time()-start)
            writer.writerow(line)


if __name__ == '__main__':
    s = time.time()
    predict("/data/renhaoye/decals_2022/model_10.model",
            "/data/renhaoye/decals_2022/out_decals/normalized_dat/")
    print(time.time()-s)