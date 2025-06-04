import torch
import torch.nn as nn
import torch.nn.functional as F
from gma import Aggregate


class IterMask(nn.Module):
    def __init__(self, input_dim=512):
        super(IterMask, self).__init__()
        self.iter_mask_0 = nn.Sequential(
            nn.Conv2d(input_dim + input_dim // 8, input_dim // 8, 3, 1, 1),
        )
        self.iter_mask_1 = nn.Sequential(
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim // 8, 3, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, tgt_sparsity):
        x = self.iter_mask_0(x)
        mhidden = x * tgt_sparsity
        x = self.iter_mask_1(mhidden)
        inc = x[:,:1]
        x = x[:,1:]
        return x, mhidden, inc

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


class GMAUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=self.args.num_heads)
        self.iter_mask = IterMask(input_dim=hidden_dim + 4 + 7)

    def forward(self, net, inp, corr, flow, attention, iter_mask=None, tgt_sparsity=None, itr_embedding= None, mhidden=None):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

        # Attentional update
        net_past = net.clone()
        net = self.gru(net, inp_cat)

        delta_flow = self.flow_head(net)

        delta_flow_wo_skip = delta_flow.clone()
        mask_wo_skip = .25 * self.mask(net)

        if iter_mask is not None:
            net = net_past * (1 - iter_mask) + net * iter_mask
            delta_flow = delta_flow * iter_mask
        bs, _, h, w = net.shape
        itr_embedding = itr_embedding.repeat(bs, 1, h, w)

        # from thop import profile
        # flops, params = profile(self.iter_mask, inputs=(torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
        #                                          tgt_sparsity,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        # exit(0)
        iter_mask_next, mhidden, inc = self.iter_mask(torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
                                                 tgt_sparsity)

        # scale mask to balence gradients
        # from thop import profile
        # print('self.mask')
        # flops, params = profile(self.mask, inputs=(net,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        # exit(0)
        mask = .25 * self.mask(net)
        return net, mask, delta_flow, iter_mask_next, mhidden, inc, mask_wo_skip, delta_flow_wo_skip

    def forward_time(self, net, inp, corr, flow, attention, iter_mask=None, tgt_sparsity=None, itr_embedding= None, mhidden=None):

        if iter_mask is None or iter_mask.view(1).item() > 0.5:

            motion_features = self.encoder(flow, corr)
            motion_features_global = self.aggregator(attention, motion_features)
            inp_cat = torch.cat([inp, motion_features, motion_features_global], dim=1)

            # Attentional update
            net = self.gru(net, inp_cat)
            delta_flow = self.flow_head(net)

            bs, _, h, w = net.shape
            itr_embedding = itr_embedding.repeat(bs, 1, h, w)
            iter_mask_next, mhidden, inc = self.iter_mask(
                torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
                tgt_sparsity)


            return net, delta_flow, iter_mask_next, mhidden
        else:
            # time_start = time.time()
            bs, _, h, w = net.shape
            itr_embedding = itr_embedding.repeat(bs, 1, h, w)
            # print(time.time()-time_start)
            delta_flow = flow * 0.0
            iter_mask_next, mhidden, inc = self.iter_mask(
                torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
                tgt_sparsity)
            # print(time.time()-time_start)

            return net, delta_flow, iter_mask_next, mhidden

    def forward_time_mask(self, net):
        mask = .25 * self.mask(net)

        return mask



