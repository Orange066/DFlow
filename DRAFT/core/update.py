import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class SpatialMask(nn.Module):
    def __init__(self, input_dim=128):
        super(SpatialMask, self).__init__()
        self.spa_mask_0 = nn.Sequential(
            nn.Conv2d(input_dim + input_dim // 8, input_dim // 8, 3, 1, 1),
        )
        self.spa_mask_1 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(input_dim // 8, input_dim // 16, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(input_dim // 16, 2, 3, 1, 1),
        )

    def forward(self, x, tgt_sparsity):
        # print('self.spa_mask_0', self.spa_mask_0)
        # print('self.spa_mask_1', self.spa_mask_1)
        x = self.spa_mask_0(x)
        x = x * tgt_sparsity
        return self.spa_mask_1(x), x

# class IterMask(nn.Module):
#     def __init__(self, input_dim=512):
#         super(IterMask, self).__init__()
#         self.iter_mask_0 = nn.Sequential(
#             nn.Conv2d(input_dim, input_dim // 8, 3, 1, 1),
#             nn.ReLU(True),
#             nn.Conv2d(input_dim // 8, input_dim // 8, 3, 1, 1),
#             nn.ReLU(True),
#             nn.AdaptiveAvgPool2d(1),
#         )
#         self.iter_mask_1_0 = nn.Sequential(
#             nn.Conv2d(input_dim // 8 + input_dim // 8 + 7, input_dim // 8-1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#         )
#         self.iter_mask_1_1 = nn.Sequential(
#             nn.Conv2d(input_dim // 8 + input_dim // 8 + 7, 1, kernel_size=1, stride=1, padding=0),
#         )
#         self.iter_mask_2 = nn.Sequential(
#             nn.Conv2d(input_dim // 8, 2, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, x, mhidden, tgt_sparsity):
#         x = self.iter_mask_0(x)
#         x_clone = x.clone()
#         x = self.iter_mask_1_0(torch.cat([x, mhidden], dim=1))
#         inc = self.iter_mask_1_1(torch.cat([x_clone, mhidden], dim=1))
#         x = x * tgt_sparsity
#         return self.iter_mask_2(torch.cat([x, inc], dim=1)), torch.cat([x, inc], dim=1), inc


# class IterMask(nn.Module):
#     def __init__(self, input_dim=512):
#         super(IterMask, self).__init__()
#         self.iter_mask_0 = nn.Sequential(
#             nn.Conv2d(input_dim, input_dim // 8, 3, 1, 1),
#             nn.ReLU(True),
#             nn.Conv2d(input_dim // 8, input_dim // 8, 3, 1, 1),
#             nn.ReLU(True),
#             nn.AdaptiveAvgPool2d(1),
#         )
#         self.iter_mask_1_0 = nn.Sequential(
#             nn.Conv2d(input_dim // 8 + input_dim // 8 + 7, input_dim // 8-1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#         )
#         self.iter_mask_1_1 = nn.Sequential(
#             nn.Conv2d(input_dim // 8 + input_dim // 8 + 7, 1, kernel_size=1, stride=1, padding=0),
#         )
#         self.iter_mask_2 = nn.Sequential(
#             nn.Conv2d(input_dim // 8, 2, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, x, mhidden, tgt_sparsity):
#         x = self.iter_mask_0(x)
#         x_clone = x.clone()
#         x = self.iter_mask_1_0(torch.cat([x, mhidden], dim=1))
#         inc = self.iter_mask_1_1(torch.cat([x_clone, mhidden], dim=1))
#         x = x * tgt_sparsity
#         return self.iter_mask_2(torch.cat([x, inc], dim=1)), torch.cat([x, inc], dim=1), inc

# old
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
        # time_start = time.time()
        x = self.iter_mask_0(x)
        mhidden = x * tgt_sparsity
        # print('1', time.time()-time_start)
        # time_start = time.time()
        x = self.iter_mask_1(mhidden)
        inc = x[:,:1]
        x = x[:,1:]
        # print('2', time.time() - time_start)
        # time_start = time.time()
        # x_clone = x.clone()
        # x = self.iter_mask_2_1(x)
        # print('3', time.time() - time_start)
        # time_start = time.time()
        # inc = self.iter_mask_2_2(x_clone)
        # print('4', time.time() - time_start)
        # time_start = time.time()
        # x = self.iter_mask_3(torch.cat([x,inc], dim=1))
        # print('5', time.time() - time_start)
        # time_start = time.time()
        return x, mhidden, inc

# class IterMask(nn.Module):
#     def __init__(self, input_dim=512):
#         super(IterMask, self).__init__()
#         self.iter_mask_0 = nn.Sequential(
#             nn.Conv2d(input_dim + input_dim // 8, input_dim // 8, 3, 1, 1),
#             nn.ReLU(True),
#         )
#         self.iter_mask_1 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#         )
#         self.iter_mask_2 = nn.Sequential(
#             nn.Conv2d(input_dim // 8, 2, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, x, tgt_sparsity):
#         time_start = time.time()
#         x = self.iter_mask_0(x)
#         mhidden = x * tgt_sparsity
#         print('1', time.time()-time_start)
#         time_start = time.time()
#         x = self.iter_mask_1(mhidden)
#         print('2', time.time() - time_start)
#         time_start = time.time()
#         x = self.iter_mask_2(x)
#         print('3', time.time() - time_start)
#         # time_start = time.time()
#         inc = x[:, :1]
#         return x, mhidden, inc

# class IterMask(nn.Module):
#     def __init__(self, input_dim=512):
#         super(IterMask, self).__init__()
#         self.iter_mask_0 = nn.Sequential(
#             nn.Conv2d(input_dim + input_dim // 8, input_dim // 8, 3, 1, 1),
#         )
#         self.iter_mask_1 = nn.Sequential(
#             nn.ReLU(True),
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(input_dim // 8, 2, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, x, tgt_sparsity):
#         # time_start = time.time()
#         x = self.iter_mask_0(x)
#         mhidden = x * tgt_sparsity
#         # print('1', time.time()-time_start)
#         # time_start = time.time()
#         x = self.iter_mask_1(mhidden)
#         inc = x[:, :1]
#         return x, mhidden, inc

# class IterMask(nn.Module):
#     def __init__(self, input_dim=512):
#         super(IterMask, self).__init__()
#         self.iter_mask_0 = nn.Sequential(
#             nn.Conv2d(input_dim + input_dim // 8, input_dim // 8, 3, 1, 1),
#         )
#         self.iter_mask_1 = nn.Sequential(
#             nn.ReLU(True),
#             nn.Conv2d(input_dim // 8, input_dim // 8, 3, 1, 1),
#             nn.ReLU(True),
#             nn.AdaptiveAvgPool2d(1),
#         )
#         self.iter_mask_2_1 = nn.Sequential(
#             nn.Conv2d(input_dim // 8, input_dim // 8, kernel_size=1, stride=1, padding=0),
#         )
#         self.iter_mask_3 = nn.Sequential(
#             nn.Conv2d(input_dim // 8, 2, kernel_size=1, stride=1, padding=0),
#         )
#         self.relu = nn.ReLU(True)
#
#     def forward(self, x, tgt_sparsity):
#         x = self.iter_mask_0(x)
#         mhidden = x * tgt_sparsity
#         x = self.iter_mask_1(mhidden)
#         x = self.iter_mask_2_1(x)
#         inc = x[:, :1]
#         # x[:,1:] = self.relu(x[:,1:])
#         x = self.iter_mask_3(torch.cat([inc, self.relu(x[:,1:])], dim=1))
#         return x, mhidden, inc


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

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
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

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

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.iter_mask = IterMask(input_dim=hidden_dim+4+7)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def get_spatial_mask_parameters(self):
        spatial_parameters = [params for params in self.iter_mask.parameters()]
        return spatial_parameters

    def forward(self, net, inp, corr, flow, iter_mask=None, tgt_sparsity=None, itr_embedding= None, mhidden=None, upsample=True):
        # time_start = time.time()
        # from thop import profile
        # print('encoder')
        # flops, params = profile(self.encoder, inputs=(flow, corr,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        # print('0_encoder', time.time() - time_start)
        # time_start = time.time()
        net_past = net.clone()
        # print('gru')
        # flops, params = profile(self.gru, inputs=(net, inp,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        net = self.gru(net, inp)
        # print('1_net', time.time() - time_start)
        # time_start = time.time()
        # print('flow_head')
        # flops, params = profile(self.flow_head, inputs=(net,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        delta_flow = self.flow_head(net)
        # print('2_flow', time.time() - time_start)

        delta_flow_wo_skip = delta_flow.clone()
        mask_wo_skip = .25 * self.mask(net)

        # scale mask to balence gradients
        if iter_mask is not None:
            net = net_past * (1-iter_mask) + net * iter_mask
            delta_flow = delta_flow * iter_mask

        # print('net', net.shape)
        # print('flow', flow.shape)
        # print('delta_flow', delta_flow.shape)
        # print('mhidden', mhidden.shape)
        # time_start = time.time()
        bs, _, h, w = net.shape
        itr_embedding = itr_embedding.repeat(bs, 1, h, w)

        # from thop import profile
        # print('iter_mask')
        # flops, params = profile(self.iter_mask, inputs=(torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1), tgt_sparsity,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        # exit(0)
        # print('net',net.shape)
        # print('flow', flow.shape)
        # print('delta_flow', delta_flow.shape)
        # print('mhidden', mhidden.shape)
        # print('itr_embedding', itr_embedding.shape)
        # iter_mask_next, mhidden = self.iter_mask(torch.cat([net, flow, delta_flow], dim=1), torch.cat([mhidden, itr_embedding], dim=1), tgt_sparsity)
        # time_start = time.time()
        iter_mask_next, mhidden, inc = self.iter_mask(torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
                                                 tgt_sparsity)
        # print('iter_mask', time.time()-time_start)


        mask = .25 * self.mask(net)
        # if spatial_mask is not None:
        #     mask = self.mask[0](net) * spatial_mask
        #     mask = self.mask[1](mask)
        #     mask = self.mask[2](mask) * spatial_mask * .25
        # else:
        #     mask = .25 * self.mask(net)
        return net, mask, delta_flow, iter_mask_next, mhidden, inc, mask_wo_skip, delta_flow_wo_skip

    def forward_inc(self, net, inp, corr, flow, iter_mask=None, tgt_sparsity=None, itr_embedding= None, mhidden=None, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        # print('net',net.shape)
        # print('flow', flow.shape)
        # print('delta_flow', delta_flow.shape)
        # print('mhidden', mhidden.shape)
        # print('itr_embedding', itr_embedding.shape)
        bs, _, h, w = net.shape
        itr_embedding = itr_embedding.repeat(bs, 1, h, w)
        iter_mask_next, mhidden, inc = self.iter_mask(torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
                                                 tgt_sparsity)

        mask = .25 * self.mask(net)
        # if spatial_mask is not None:
        #     mask = self.mask[0](net) * spatial_mask
        #     mask = self.mask[1](mask)
        #     mask = self.mask[2](mask) * spatial_mask * .25
        # else:
        #     mask = .25 * self.mask(net)
        return net, mask, delta_flow, iter_mask_next, mhidden, inc

    def forward_time(self, net, inp, corr, flow, iter_mask=None, tgt_sparsity=None, itr_embedding= None, mhidden=None, upsample=True, output = False):

        if iter_mask is None or iter_mask.view(1).item() > 0.5:
            # time_start = time.time()
            # from thop import profile
            # print('encoder')
            # flops, params = profile(self.encoder, inputs=(flow, corr,))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 1) + 'M')
            # print('*' * 60)
            motion_features = self.encoder(flow, corr)
            inp = torch.cat([inp, motion_features], dim=1)

            # print('encoder')
            # flops, params = profile(self.gru, inputs=(net,inp,))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 1) + 'M')
            # print('*' * 60)
            net = self.gru(net, inp)
            # print('encoder')
            # flops, params = profile(self.flow_head, inputs=(net,))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 1) + 'M')
            # print('*' * 60)
            delta_flow = self.flow_head(net)
            # print('2_flow', time.time() - time_start)

            # bs, _, h, w = net.shape
            # itr_embedding = itr_embedding.repeat(bs, 1, h, w)
            # time_start = time.time()
            # from thop import profile
            # print('iter_mask')
            # flops, params = profile(self.iter_mask, inputs=(
            # torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1), tgt_sparsity,))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 1) + 'M')
            # print('*' * 60)
            # exit(0)
            # print('iter', itr_embedding.shape)


            bs, _, h, w = net.shape
            itr_embedding = itr_embedding.repeat(bs, 1, h, w)
            iter_mask_next, mhidden, inc = self.iter_mask(
                torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
                tgt_sparsity)


            # print('3_iter', time.time() - time_start)

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

    # def forward_time(self, net, inp, corr, flow, iter_mask=None, tgt_sparsity=None, itr_embedding= None, mhidden=None, upsample=True):
    #     motion_features = self.encoder(flow, corr)
    #     inp = torch.cat([inp, motion_features], dim=1)
    #
    #     net = self.gru(net, inp)
    #     delta_flow = self.flow_head(net)
    #
    #     bs, _, h, w = net.shape
    #     itr_embedding = itr_embedding.repeat(bs, 1, h, w)
    #     iter_mask_next, mhidden, _ = self.iter_mask(
    #         torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
    #         tgt_sparsity)
    #     iter_mask_next = (iter_mask_next[:, :1, ...] > iter_mask_next[:, 1:, ...]).float()
    #     return net, delta_flow, iter_mask_next, mhidden

    def forward_time_mask(self, net):
        mask = .25 * self.mask(net)

        return mask

    def forward_time_iter_mask(self, net, flow, mhidden, itr_embedding, tgt_sparsity):
        bs, _, h, w = net.shape
        itr_embedding = itr_embedding.repeat(bs, 1, h, w)
        iter_mask_next, mhidden, _ = self.iter_mask(
            torch.cat([net, flow, flow*0.0, mhidden, itr_embedding], dim=1),
            tgt_sparsity)
        iter_mask_next = (iter_mask_next[:, :1, ...] > iter_mask_next[:, 1:, ...]).float()
        return iter_mask_next, mhidden

    # def forward_time(self, net, inp, corr, flow, upsample=True):
    #     # time_start = time.time()
    #     motion_features = self.encoder(flow, corr)
    #
    #     inp = torch.cat([inp, motion_features], dim=1)
    #     net = self.gru(net, inp)
    #     delta_flow = self.flow_head(net)
    #     # print('flow', time.time()-time_start)
    #     return net, delta_flow
    #
    # def forward_time_mask(self, net):
    #     mask = .25 * self.mask(net)
    #
    #     return mask




