from GRUs.my_convgru import ConvGRU
import torch
from torch.autograd import Variable
# Generate a ConvGRU with 3 cells
# input_size and hidden_sizes reflect feature map depths.
# Height and Width are preserved by zero padding within the module.
model = ConvGRU(input_size=8, hidden_sizes=[32,64,16],
                  kernel_sizes=[3, 5, 3], n_layers=3)

x = Variable(torch.FloatTensor(1,2,64,64))
output = model(x)

# output is a list of sequential hidden representation tensors
print(type(output)) # list

# final output size
print(output[-1].size()) # torch.Size([1, 16, 64, 64])