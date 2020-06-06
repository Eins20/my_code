import torch
from unet import Unet
import argparse
from torch import nn,optim
from torchvision.transforms import transforms
from dataset import LiverDataset
from torch.utils.data import DataLoader
import numpy as np
from edge import draw_edge
import os
import cv2 as cv
from PIL import Image
import time
from edge import remove_small

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

y_transforms = transforms.ToTensor()
import sys
def train_model(model,criterion,optimizer,dataload,num_epochs=30):
    last_loss = sys.maxsize
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        epoch_loss = 0
        step = 0
        for x,y,_ in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            #zero the parameter gradients
            optimizer.zero_grad()
            #forward
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("%d/%d,train_loss:%0.3f" %(step,(dt_size-1)//dataload.batch_size+1,loss.item()))
        print("epoch %d loss:%0.3f" %(epoch,epoch_loss/step))
        if abs(last_loss-epoch_loss)<1e-5:break
        last_loss = epoch_loss
    torch.save(model.state_dict(),'weights1000_%d.pth'%epoch)
    return model

def train(args):
    model = Unet(3,1).to(device)
    #begin add
    # checkpoint = torch.load("./weights_19.pth",map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    #end add
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("./data/train",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    train_model(model,criterion,optimizer,dataloaders)

def test(args):
    model = Unet(3,1)
    model.load_state_dict(torch.load(args.ckpt))
    liver_dataset = LiverDataset("/home/ices/work/tzh/predrnn/results/my_data_predrnn/1050/1",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset,batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x,name in dataloaders:
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            img_y = np.asarray(img_y)
            copy = img_y
            # print(len(copy[copy<0.5]))
            copy[copy>0.5] = int(255)
            copy[copy<=0.5] = int(0)
            copy = copy.astype(np.int16)
            copy = cv.resize(copy, (64,64))

            cv.imwrite(os.path.join("./data/my_result", name[0] + ".png"),copy)
            print(name[0])
            copy = cv.imread(os.path.join("./data/my_result", name[0] + ".png"),0)

            # raw_img = cv.imread(os.path.join("../Unet/raw_images",name[0]+".png"))
            # cv.imwrite(os.path.join("./data/train",name[0]+".png"),raw_img)
            # kernel = np.ones((7, 7), np.uint8)
            # copy = cv.morphologyEx(copy, cv.MORPH_CLOSE, kernel)
            # copy = remove_small(copy, 100)
            # copy = cv.GaussianBlur(copy,(3,3),0)
            # cv.imwrite(os.path.join("./data/result",name+"hihi.png"),file)
            # cv.imwrite(os.path.join("./data/result",name+"_mask.png"),file)

            edges = cv.Canny(copy, 50, 150)
            # print(len(edges[edges!=0]))
            cv.imwrite(os.path.join("./data/my_result", name[0] + ".png"), edges)
            draw_edge(name[0])

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--action",type=str,help="train or test",default="test")
    parse.add_argument("--batch_size",type=int,default=4)
    parse.add_argument('--ckpt',type=str,help="the path of model weght file",default='weights1000_29.pth')
    args = parse.parse_args()
    # test(args)

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)