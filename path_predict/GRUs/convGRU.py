from torch import nn
from GRUs.ConvGRUCell import ConvGRUCell

#encode
def conv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias)]
    # layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def downsmaple(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    # torch.cat((x, x, x), dim = 0)
    # the downsample layer input is last rnn output,like:output[-1]
    ret = conv2_act(inplanes, out_channels, kernel_size, stride, padding, bias)
    return ret


class Encoder(nn.Module):
    def __init__(self, inplanes):
        super(Encoder, self).__init__()
        # self.num_seqs = num_seqs
        self.conv1_act = conv2_act(inplanes, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv2_act = conv2_act(8, out_channels=16, kernel_size=4, stride=2, padding=1, bias=True)
        num_filter = [64, 192, 192]
        kernel_size_l = [7,7,5]
        rnn_block_num = len(num_filter)
        stack_num = [2, 3, 3]
        encoder_rnn_block_states = []
        self.rnn1_1 = ConvGRUCell(input_size=16, hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter[0], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn1_2_h = None
        self.downsample1 = downsmaple(inplanes=num_filter[0], out_channels=num_filter[1], kernel_size=4, stride=2,
                                      padding=1)

        self.rnn2_1 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_1_h = None
        self.rnn2_2 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_2_h = None
        self.rnn2_3 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_3_h = None

        self.downsample2 = downsmaple(inplanes=num_filter[1], out_channels=num_filter[2], kernel_size=5, stride=3,
                                      padding=1)

        self.rnn3_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_1_h = None
        self.rnn3_2 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_2_h = None
        self.rnn3_3 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn3_3_h = None


    def init_h0(self):
        self.rnn1_1_h = None
        self.rnn1_2_h = None
        self.rnn2_1_h = None
        self.rnn2_2_h = None
        self.rnn2_3_h = None
        self.rnn3_1_h = None
        self.rnn3_2_h = None
        self.rnn3_3_h = None
    def forward(self, data):
        # print data.size()
        data = self.conv1_act(data)
        # print data.size()

        # data = self.conv2_act(data)
        # print data.size()
        self.rnn1_1_h = self.rnn1_1(data, self.rnn1_1_h)
        self.rnn1_2_h = self.rnn1_2(self.rnn1_1_h, self.rnn1_2_h)
        # print self.rnn1_2_h.size()
        # data = torch.cat(self.rnn1_2_h, dim=0)
        # print data.size()
        data = self.downsample1(self.rnn1_2_h)
        # print data.size()
        self.rnn2_1_h = self.rnn2_1(data, self.rnn2_1_h)

        self.rnn2_2_h = self.rnn2_2(self.rnn2_1_h, self.rnn2_2_h)

        self.rnn2_3_h = self.rnn2_3(self.rnn2_2_h, self.rnn2_3_h)
        # print self.rnn2_3_h.size()
        # data = torch.cat(*self.rnn2_3_h, dim=0)
        data = self.downsample2(self.rnn2_3_h)
        # print data.size()
        self.rnn3_1_h = self.rnn3_1(data, self.rnn3_1_h)

        self.rnn3_2_h = self.rnn3_2(self.rnn3_1_h, self.rnn3_2_h)

        self.rnn3_3_h = self.rnn3_3(self.rnn3_2_h, self.rnn3_3_h)
        # print self.rnn3_3_h.size()
        return self.rnn2_3_h

#forecast
def deconv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.ConvTranspose2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def upsmaple(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    # torch.cat((x, x, x), dim = 0)
    # the downsample layer input is last rnn output,like:output[-1]
    ret = deconv2_act(inplanes, out_channels, kernel_size, stride, padding, bias)
    return ret

class Forecaster(nn.Module):
    def __init__(self):
        super(Forecaster, self).__init__()

        num_filter = [64, 192, 192]
        kernel_size_l = [7, 7, 5]
        self.rnn1_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_2_h = None
        self.rnn1_3 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_3_h = None
        #
        self.upsample1 = upsmaple(inplanes=num_filter[2], out_channels=num_filter[2], kernel_size=5, stride=3,
                                      padding=1)

        self.rnn2_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_1_h = None
        self.rnn2_2 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_2_h = None
        self.rnn2_3 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_3_h = None

        self.upsample2 = upsmaple(inplanes=num_filter[1], out_channels=num_filter[1], kernel_size=4, stride=2,
                                      padding=1 )

        self.rnn3_1 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn3_1_h = None
        self.rnn3_2 = ConvGRUCell(input_size=num_filter[0], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn3_2_h = None

        self.deconv1 = deconv2_act(inplanes =num_filter[0] , out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.deconv2 = deconv2_act(inplanes=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.conv_final = conv2_act(inplanes=16, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.conv_pre = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)

    def set_h0(self,encoder):
        self.rnn1_1_h = encoder.rnn3_3_h
        self.rnn1_2_h = encoder.rnn3_2_h
        self.rnn1_3_h = encoder.rnn3_1_h
        self.rnn2_1_h = encoder.rnn2_3_h
        self.rnn2_2_h = encoder.rnn2_2_h
        self.rnn2_3_h = encoder.rnn2_1_h
        self.rnn3_1_h = encoder.rnn1_2_h
        self.rnn3_2_h = encoder.rnn1_1_h


    def forward(self,data):
        # print data.size()
        self.rnn1_1_h = self.rnn1_1(data,self.rnn1_1_h)
        self.rnn1_2_h = self.rnn1_1(self.rnn1_1_h, self.rnn1_2_h)
        self.rnn1_3_h = self.rnn1_1(self.rnn1_2_h, self.rnn1_3_h)
        # # print self.rnn1_3_h.size()
        data = self.upsample1(self.rnn1_3_h)
        # print data.size()
        self.rnn2_1_h = self.rnn2_1(data, self.rnn2_1_h)

        self.rnn2_2_h = self.rnn2_2(self.rnn2_1_h, self.rnn2_2_h)

        self.rnn2_3_h = self.rnn2_3(self.rnn2_2_h, self.rnn2_3_h)
        # print self.rnn2_3_h.size()
        data = self.upsample2(self.rnn2_3_h)
        # print data.size()
        self.rnn3_1_h = self.rnn3_1(data, self.rnn3_1_h)

        self.rnn3_2_h = self.rnn3_2(self.rnn3_1_h, self.rnn3_2_h)
        # print self.rnn3_2_h.size()
        data = self.deconv1(self.rnn3_2_h)
        # print data.size()
        # data = self.deconv2(data)
        # print data.size()
        data = self.conv_final(data)
        # print data.size()
        pre_data = self.conv_pre(data)
        # print pre_data.size()

        return pre_data


class EFModel(nn.Module):
    def __init__(self, inplanes, input_num_seqs, output_num_seqs):
        super(EFModel, self).__init__()
        self.input_num_seqs = input_num_seqs
        self.output_num_seqs = output_num_seqs
        self.encoder = Encoder(inplanes=inplanes)
        self.forecaster = Forecaster()

    def forward(self, data):
        self.encoder.init_h0()
        self.encoder(data)
        self.forecaster.set_h0(self.encoder)
        pre_data = self.forecaster(None)
            # print h_next.size()

        return pre_data
