from torch.utils.data import Dataset
from preprocess.data_handle import *


class DecalsDataset(Dataset):
    def __init__(self, annotations_file, transform):
        with open(annotations_file, "r") as file:
            imgs = []
            for line in file:
                line = line.strip("\n")
                line = line.rstrip("\n")
                words = line.split()
                imgs.append((words[0], str(words[1])))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = load_img(path, transform=self.transform)
        # if 1 in np.isnan(img):
        #     print(img)
        return np.nan_to_num(img), np.array(int(label))

    def __len__(self):
        return len(self.imgs)
