import torch
from torch import nn,optim
from load_data import load_data
from GRUs.convGRU import EFModel

input_num_seqs = 10
output_num_seqs = 10
hidden_size = 3
input_channels_img = 1
output_channels_img = 1
size_image = 240
max_epoch = 12
cuda_flag = True
kernel_size = 3
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = EFModel(input_channels_img,input_num_seqs,output_num_seqs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=(0.001), momentum=0.9, weight_decay=0.005)
    train_data,test_data = load_data()
    for epoch in range(max_epoch):
        print('Epoch {}/{}'.format(epoch,max_epoch-1))
        print('-'*10)
        epoch_loss = 0
        step = 0
        for x,y in train_data:
            x = x.float()
            y = y.float()
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            # print("inputs",inputs)
            # print(inputs.type())
            outputs = model(inputs)
            outputs = outputs.permute(0,2,3,1)
            outputs = torch.squeeze(outputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("epoch %d loss:%0.7f" %(epoch,epoch_loss/step))

    torch.save(model.state_dict(),'weights_gray_%d.pth'%epoch)
    print("inputs:",inputs)
    print("outputs:",labels)
    test = model(input)
    # test = np.array([[502., 323.],[510., 329.], [513., 327.],[519., 329.],[529., 323.]])
    print("test:",model(test))
    return model

if __name__ == '__main__':
    train()