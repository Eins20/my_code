import torch
from torch import nn,optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2 as cv

def crop(img):
    img = np.array(img)
    min_x,min_y,max_x,max_y = 0,0,0,0
    flag = False
    for x in range(len(img)):
        temp = img[x]
        if (not flag) and temp.max()>0:
            min_x = x
            flag = True
        if (flag and temp.max()==0):
            max_x = x
            break
    flag = False
    for y in range(len(img[0])):
        temp = img[:,y]
        if (not flag) and temp.max()>0:
            min_y = y
            flag = True
        if (flag and temp.max()==0):
            max_y = y
            break
    # print(min_x,max_x,min_y,max_y)
    if max_x-min_x>128 or max_y-min_y>128:
        print("ERROR: max_x-min_x>128 or max_y-min_y>128！")
        exit()
    # print(min_x,min_y)
    sample = np.array(img)[min_x:min_x+128,min_y:min_y+128].flatten()
    # print(sample,sample.shape)
    points = np.append(np.array([min_x,min_y]),sample)
    # print(points)
    return points


# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def make_dataset(root):
    datas = []
    for dir in os.listdir(root):
        if not os.path.isdir(os.path.join(root,dir)):continue
        samples = []
        for file in os.listdir(os.path.join(root,dir)):
            raw_img = cv.imread(os.path.join(root,dir,file),0)
            raw_img[raw_img!=0] = 1
            sample = crop(raw_img)
            # img = x_transforms(img)
            samples.append(sample)
        if len(samples)<6: continue
        x1,x2,x3,x4,x5 = samples[:5]
        for sample in samples[5:]:
            # if sample[0]==1 and sample[1]==1: break
            datas.append((x1,x2,x3,x4,x5,sample))
            x1, x2, x3, x4, x5 = x2,x3,x4,x5,sample
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


class GRU_net(nn.Module):

    def __init__(self, input_size):
        super(GRU_net, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Sequential(
            nn.Linear(128, 128*128+2)
        )

    def forward(self, x):
        # print("didi")
        # print(x)
        temp = self.rnn(x, None)
        r_out, (h_n, h_c) = temp  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        for o in out:
            o[2:] = torch.sigmoid(o[2:])
        # print("out",out)
        # out_sig = torch.sigmoid(out)
        # print(out.shape)
        return out


def train_model(model,criterion,optimizer,dataload,num_epochs=2000):
    inputs = None
    labels = None
    best_model = None
    best_loss = 1000000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)
        epoch_loss = 0
        step = 0
        for x,y in dataload:
            # print(x,y)
            # print(x.type())
            x = x.float()
            # print("x_new.type():",x.type())
            y = y.float()
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # print("inputs",inputs.size())
            # print("labels",labels.size())
            optimizer.zero_grad()
            # print("inputs",inputs)
            # print(inputs.type())
            outputs = model(inputs)
            # print("outputs",outputs)
            # print("outputs",outputs.size())
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("epoch %d loss:%0.7f" %(epoch,epoch_loss/step))
        if abs(epoch_loss)<abs(best_loss):
            best_loss = epoch_loss
            best_model = model
            if abs(best_loss)<1e-6: break
    torch.save(best_model.state_dict(),'weights_line_%d.pth'%epoch)
    print("inputs:",inputs)
    print("outputs:",labels)
    test = inputs
    # test = np.array([[502., 323.],[510., 329.], [513., 327.],[519., 329.],[529., 323.]])
    print("test:",model(test))
    return model


def test_model(model,criterion,dataload):
    test_loss = 0
    for x,y in dataload:
        inputs = x.to(device)
        labels = y.to(device)
        inputs = inputs.float()
        labels = labels.float()
            # print("inputs",inputs.size())
            # print("labels",labels.size())
        optimizer.zero_grad()
        outputs = model(inputs)
        out = model(inputs)
        for o in out:
            p0 = o[:2]
            img = torch.squeeze(o[2:]).cpu().detach().numpy()
            img.resize([128,128])
            img[img>0.5] = 255
            img[img<=0.5] = 0
            cv.imwrite("test.png",img)
        # print("inputs:", inputs)
        # print(labels[0][0])
        # cv.imwrite("label.png",labels[0][0])
        # cv.imwrite("test.png",model(inputs)[0][0])
        print("label:", labels)
        print("test:", p0,img)
            # print("outputs",outputs)
            # print("outputs",outputs.size())
        loss = criterion(outputs,labels)
        test_loss += loss.item()
        print("total_loss:%0.3f" % (test_loss))

model = GRU_net(128*128+2).to(device)

def loss_func(label,output):
    loss = 0
    for l,o in zip(label,output):
        loss+=nn.L1Loss()(l[:2],o[:2])
        loss+=nn.BCELoss()(l[2:],o[2:])
    return loss

criterion = loss_func
optimizer = optim.Adam(model.parameters())
datas = make_dataset("./skeleton_datas")

np.random.shuffle(datas)
train_data = datas[:int(len(datas)*0.8)]
test_data = datas[int(len(datas)*0.8):]
# np.savetxt(os.path.join("./datas/train.txt"),np.around(np.asarray(train_data), decimals=1), fmt='%s',newline='\n')
# np.savetxt(os.path.join("./datas/test.txt"),np.around(np.asarray(test_data), decimals=1), fmt='%s',newline='\n')
liver_dataset = LiverDataset(train_data)
dataloaders = DataLoader(liver_dataset, batch_size=2, shuffle=True, num_workers=4)
model = train_model(model, criterion, optimizer, dataloaders)

liver_dataset = LiverDataset(test_data)
dataloaders_test = DataLoader(liver_dataset, batch_size=1, shuffle=True, num_workers=1)
test_model(model,criterion,dataloaders_test)
