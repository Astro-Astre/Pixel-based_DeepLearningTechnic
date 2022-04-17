from train import *
from train import SAVE_PATH, BATCH_SIZE
from torch.utils.data import DataLoader
from decals_dataset import *
from preprocess.data_handle import *
import torchvision.transforms as transforms

transfer = transforms.Compose([
    transforms.ToTensor(),
])


def predict(model_name):
    model = torch.load(model_name)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    acces = 0
    with torch.no_grad():
        for X, label in validation_loader:
            label = torch.as_tensor(label, dtype=torch.long)
            X, label = X.to(device), label.to(device)
            test_out = model(X)
            _, pred = test_out.max(1)
            num_correct = (pred == label).sum()
            acc = int(num_correct) / X.shape[0]
            acces += acc
        print(acces / len(validation_loader))


if __name__ == "__main__":
    TRAININGSET_TXT = SAVE_PATH + "training_multimask_7.txt"
    TESTSET_TXT = SAVE_PATH + "test_multimask_7.txt"
    VALIDATIONSET_TXT = SAVE_PATH + "validation_multimask_7.txt"

    train_data = DecalsDataset(annotations_file=TRAININGSET_TXT, transform=transfer)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=20, pin_memory=True)
    test_data = DecalsDataset(annotations_file=TESTSET_TXT, transform=transfer)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=20, pin_memory=True)
    validation_data = DecalsDataset(annotations_file=VALIDATIONSET_TXT, transform=transfer)
    validation_loader = DataLoader(dataset=validation_data, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=20, pin_memory=True)
    mkdir(SAVE_PATH + "model/")
    # modelPkg_name = SAVE_PATH + "model_auto/" + "densenet264_focal_2_5class/"
    modelPkg_name = SAVE_PATH + "trained_model/" + "densenet_multiMask/"
    trainModel(modelPkg_name, flag=False, last_epoch=17, train_loader=train_loader, test_loader=test_loader,
               validation_loader=validation_loader)
