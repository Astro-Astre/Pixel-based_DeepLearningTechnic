from preprocess.data_handle import *
from train import *
from torch.utils.data import DataLoader
from decals_dataset import *
TESTSET_TXT = SAVE_PATH + "test_auto.txt"
VALIDATIONSET_TXT = SAVE_PATH + "validation_auto_7.txt"

transfer = transforms.Compose([
    transforms.ToTensor(),
])

validation_data = DecalsDataset(annotations_file=VALIDATIONSET_TXT, transform=transfer)
validation_loader = DataLoader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=32,
                               pin_memory=True)


def predict(model_name, valid: bool = True, t=0):
    model = torch.load(model_name)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    cf_metrics(validation_loader, model,
                          "/data/renhaoye/decals_2022/model_%d.png" % t, valid)


if __name__ == '__main__':
    predict("/data/renhaoye/decals_2022/model_11.model", t=11)
