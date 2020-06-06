import torch
from torch import nn,optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2 as cv

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataset(root):
    datas = []
    for file in os.listdir(root):
        if not file.endswith('.txt'):continue
        samples = np.loadtxt(os.path.join(root,file))
        if len(samples)<6: continue
        x1,x2,x3,x4,x5 = samples[:5]
        print("x1:",x1)
        for sample in samples[5:]:
            if sample[0]==1 and sample[1]==1: break
            datas.append((x1,x2,x3,x4,x5,sample))
            x1, x2, x3, x4, x5 = x2,x3,x4,x5,sample
    print("datas",datas[0])
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
            nn.Linear(128, 5)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        # print(out.shape)
        return out


def train_model(model,criterion,optimizer,dataload,num_epochs=7000):
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
            x = x.float()
            y = y.float()
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # print("inputs",inputs.size())
            # print("labels",labels.size())
            optimizer.zero_grad()
            outputs = model(inputs)
            # print("outputs",outputs)
            # print("outputs",outputs.size())
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("epoch %d loss:%0.3f" %(epoch,epoch_loss/step))
        if epoch_loss<best_loss:
            best_loss = epoch_loss
            best_model = model
            if best_loss<100: break
    torch.save(best_model.state_dict(),'weights_elli_%d.pth'%epoch)
    print("inputs:",inputs)
    print("outputs:",labels)
    test = inputs
    # test = np.array([[502., 323.],[510., 329.], [513., 327.],[519., 329.],[529., 323.]])
    print("test:",model(test))
    return model


def test_model(model,criterion,dataload):
    test_loss = 0
    for x,y in dataload:
        x = x.float()
        y = y.float()
        inputs = x.to(device)
        labels = y.to(device)
            # print("inputs",inputs.size())
            # print("labels",labels.size())
        optimizer.zero_grad()
        outputs = model(inputs)
        print("inputs:", inputs)
        print("outputs:", labels)
        print("test:", model(inputs))
            # print("outputs",outputs)
            # print("outputs",outputs.size())
        loss = criterion(outputs,labels)
        test_loss += loss.item()
        print("total_loss:%0.3f" % (test_loss))

model = GRU_net(5).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
datas = make_dataset("../datas")

np.random.shuffle(datas)
train_data = datas[:int(len(datas)*0.8)]
test_data = datas[int(len(datas)*0.8):]
# np.savetxt(os.path.join("./datas/train.txt"),np.around(np.asarray(train_data), decimals=1), fmt='%s',newline='\n')
# np.savetxt(os.path.join("./datas/test.txt"),np.around(np.asarray(test_data), decimals=1), fmt='%s',newline='\n')
liver_dataset = LiverDataset(train_data)
dataloaders = DataLoader(liver_dataset, batch_size=4, shuffle=True, num_workers=4)
model = train_model(model, criterion, optimizer, dataloaders)

liver_dataset = LiverDataset(test_data)
dataloaders_test = DataLoader(liver_dataset, batch_size=4, shuffle=True, num_workers=4)
test_model(model,criterion,dataloaders_test)
