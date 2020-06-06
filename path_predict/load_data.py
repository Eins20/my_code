import os
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def make_dataset():
    files = os.listdir("/home/ices/work/tzh/my_unet/data/result")
    files = sorted(files)[:85]
    names = [file.split('.')[0] for file in files]
    files = [os.path.join("/extend/14-17_2500_radar",name[12:14]+"_2500_radar",
                        name+".ref") for name in names]

    datas = []
    samples = []
    for file in files:
        # print(file)
        # raw_img = cv.imread(file)
        ref_f = np.fromfile(file, dtype=np.uint8).reshape(700, 900)
        ref_f[ref_f >= 80] = 0
        ref_f[ref_f <= 45] = 0
        raw_img = ref_f
        sample = cv.resize(raw_img,(128,128))
        samples.append(sample)

    while len(samples)>5:
        x1,x2,x3,x4,x5,label = samples[:6]
            # if sample[0]==1 and sample[1]==1: break
        datas.append((x1,x2,x3,x4,x5,label))
        samples = samples[6:]
    return datas


class LiverDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas
        self.train = self.datas[:int(len(self.datas)*0.8)]
        self.test = self.datas[int(len(self.datas)*0.8):]

    def __getitem__(self, index):
        x1,x2,x3,x4,x5,y = self.datas[index]
        y = np.asarray(y)
        x = np.array([x1,x2,x3,x4,x5])
        x = torch.from_numpy(x)
        # print("x",x)
        # print("hihi",x.ndim)
        return x,y

    def __len__(self):
        return len(self.datas)

def load_data():
    datas = make_dataset()
    np.random.shuffle(datas)
    train_data = datas[:int(len(datas) * 0.8)]
    test_data = datas[int(len(datas) * 0.8):]
    liver_dataset = LiverDataset(train_data)
    train_dataloaders = DataLoader(liver_dataset, batch_size=4, shuffle=True, num_workers=4)
    liver_dataset = LiverDataset(test_data)
    test_dataloaders_test = DataLoader(liver_dataset, batch_size=4, shuffle=True, num_workers=4)
    return train_dataloaders,test_dataloaders_test