import torch
from torchsummary import summary
from torch import nn,optim
from torchvision import transforms,models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import numpy as np
import cv2 as cv

SEQ_SIZE = 5
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class EncoderMUG2d_LSTM(nn.Module):
    def __init__(self, input_nc=1, encode_dim=1024, lstm_hidden_size=1024, seq_len=SEQ_SIZE, num_lstm_layers=1,
                 bidirectional=False):
        super(EncoderMUG2d_LSTM, self).__init__()
        self.seq_len = seq_len
        self.num_directions = 2 if bidirectional else 1
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        # 3*128*128
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 32, 4, 2, 1),  # 32*64*64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32*63*63
            nn.Conv2d(32, 64, 4, 2, 1),  # 64*32*32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64*31*31
            nn.Conv2d(64, 128, 4, 2, 1),  # 128*16*16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # 256*8*8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),  # 512*4*4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 4, 2, 1),  # 512*2*2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1),  # 1024*1*1
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.fc = nn.Linear(1024, encode_dim)
        self.lstm = nn.LSTM(encode_dim, encode_dim, batch_first=True)

    def init_hidden(self, x):
        batch_size = x.size(0)
        h = x.data.new(
            self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        c = x.data.new(
            self.num_directions * self.num_lstm_layers, batch_size, self.lstm_hidden_size).zero_()
        return Variable(h), Variable(c)

    def forward(self, x):
        # x.shape [batchsize,seqsize,3,128,128]
        B = x.size(0)
        x = x.view(B * SEQ_SIZE, 1, 128, 128)  # x.shape[batchsize*seqsize,3,128,128]
        # [batchsize*seqsize, 3, 128, 128] -> [batchsize*seqsize, 1024,1,1]
        x = self.encoder(x)
        # [batchsize * seqsize, 1024, 1, 1]-> [batchsize*seqsize, 1024]
        x = x.view(-1, 1024)
        # [batchsize * seqsize, 1024]
        x = self.fc(x)
        # [batchsize , seqsize ,1024]
        x = x.view(-1, SEQ_SIZE, x.size(1))
        h0, c0 = self.init_hidden(x)
        # print(x.size())
        output, (hn, cn) = self.lstm(x, None)
        # output, (hn, cn) = self.lstm(x, (h0, c0))
        return hn


class DecoderMUG2d(nn.Module):
    def __init__(self, output_nc=1, encode_dim=1024):  # output size: 64x64
        super(DecoderMUG2d, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(encode_dim, 1024 * 1 * 1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4),  # 512*4*4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, stride=2),  # 256*10*10
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4),  # 128*13*13
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, stride=2),  # 64*28*28
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4),  # 32*31*31
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, stride=2),  # 16*64*64
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, output_nc, 4, stride=2, padding=1),  # 3*128*128
            # nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.project(x)
        x = x.view(-1, 1024, 1, 1)
        decode = self.decoder(x)
        return decode


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.n1 = EncoderMUG2d_LSTM()
        self.n2 = DecoderMUG2d()

    def forward(self, x):
        output = self.n1(x)
        output = self.n2(output)  # B*3*128*128
        return output

def train_model(model,criterion,optimizer,dataload,num_epochs=8000):
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
            # x = x.permute(0,1,2,3)
            # print("x.size()",x.size())
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
            outputs = outputs.permute(0,2,3,1)
            outputs = torch.squeeze(outputs)
            # print("output",outputs.size())
            # print("label",labels.size())
            # print("outputs",outputs)
            # print("outputs",outputs.size())
            # loss = 0
            # for y,y_ in zip(outputs,labels):
            #     loss+=(y-y_)**2
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("epoch %d loss:%0.7f" %(epoch,epoch_loss/step))
        if abs(epoch_loss)<abs(best_loss):
            best_loss = epoch_loss
            best_model = model
            if abs(best_loss)<1e-6: break
    torch.save(best_model.state_dict(),'weights_gray_%d.pth'%epoch)
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
            # img = torch.squeeze(o).cpu().detach().numpy()
            img = o.cpu().detach().numpy()
            img = img[0]
            cv.imwrite("test_.png",img)
            print(img.shape)
            img.resize([900,700])
            cv.imwrite("test.png",img)
        # print("inputs:", inputs)        # print(labels[0][0])
        # cv.imwrite("label.png",labels[0][0])
        # cv.imwrite("test.png",model(inputs)[0][0])
        # print("label:", labels)
        # print("test:", p0,img)
            # print("outputs",outputs)
            # print("outputs",outputs.size())
        loss = criterion(outputs,labels)
        test_loss += loss.item()
        print("total_loss:%0.3f" % (test_loss))

model = net().to(device)
# model.load_state_dict(torch.load("weights_gray_1999.pth"))
# print(summary(model,(5,3,128,128)))
# model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
datas = make_dataset()

np.random.shuffle(datas)
train_data = datas[:int(len(datas)*0.8)]
test_data = datas[int(len(datas)*0.8):]
# np.savetxt(os.path.join("./datas/train.txt"),np.around(np.asarray(train_data), decimals=1), fmt='%s',newline='\n')
# np.savetxt(os.path.join("./datas/test.txt"),np.around(np.asarray(test_data), decimals=1), fmt='%s',newline='\n')
liver_dataset = LiverDataset(train_data)
dataloaders = DataLoader(liver_dataset, batch_size=4, shuffle=True, num_workers=4)
model = train_model(model, criterion, optimizer, dataloaders)

liver_dataset = LiverDataset(datas[-1:])
dataloaders_test = DataLoader(liver_dataset, batch_size=1, shuffle=True, num_workers=1)
test_model(model,criterion,dataloaders_test)
