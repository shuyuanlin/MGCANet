import torch
import torch.nn as nn
from loss import batch_episym
import torch.nn.functional as F
from torch.nn import Conv2d

class SE(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):  # num_channels=64
        super(SE, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.conv0 = Conv2d(num_channels, num_channels,
                            kernel_size=1, stride=1, bias=True)
        self.in0 = nn.InstanceNorm2d(num_channels)
        self.bn0 = nn.BatchNorm2d(num_channels)
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        x = self.in0(input_tensor)
        x = self.bn0(x)
        x = self.relu(x)  # b,128,2000,1
        input_tensor = self.conv0(x)
        squeeze_tensor = input_tensor.view(
            batch_size, num_channels, -1).mean(dim=2)  # 对每个通道求平均值  b,128
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))  # b,64
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))  # b,128
        a, b = squeeze_tensor.size()  # a:batch_size, b:128
        # b,128,2000,1    b,128,1,1---->b,128,2000,1
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class MBSE(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, bottleneck_width=64):  # planes=64
        super(MBSE, self).__init__()
        SE_channel = int(planes * (bottleneck_width / 64.))
        self.shot_cut = None
        if planes * 2 != inplanes:
            self.shot_cut = nn.Conv2d(inplanes, planes * 2, kernel_size=1)
        self.conv1 = nn.Conv2d(inplanes, SE_channel, kernel_size=1, bias=True)
        self.in1 = nn.InstanceNorm2d(inplanes, eps=1e-5)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = SE(SE_channel)
        self.conv3 = nn.Conv2d(SE_channel, planes * 2, kernel_size=1, bias=True)
        self.in3 = nn.InstanceNorm2d(SE_channel, eps=1e-5)
        self.bn3 = nn.BatchNorm2d(SE_channel)
        self.conv_branch1_1 = nn.Sequential(
            nn.InstanceNorm2d(inplanes, eps=1e-3),
            nn.BatchNorm2d(inplanes),
            nn.GELU(),
            nn.Conv2d(inplanes, planes * 2, kernel_size=1),
        )
        self.conv_merge = nn.Sequential(
            nn.InstanceNorm2d(planes * 2, eps=1e-3),
            nn.BatchNorm2d(planes * 2),
            nn.GELU(),
            nn.Conv2d(planes * 2, planes * 2, kernel_size=1),
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.in1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.in3(out)
        out = self.bn3(out)
        out = self.conv3(out)
        branch_out = self.conv_branch1_1(x)
        # branch_out = self.conv_branch1_2(branch_out)
        if self.shot_cut:
            residual = self.shot_cut(x)
        else:
            residual = x
        out = out + branch_out + residual
        out = self.conv_merge(out)
        return out

def down_sampling(x, indices, features=None):
    B, _, N, _ = x.size()
    # 取t
    indices = indices[:, :int(N * 0.5)]
    indices = indices.view(B, 1, -1, 1)

    with torch.no_grad():
        x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))
    feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))
    return x_out, feature_out


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx[:, :, :]

class MLPs(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_out + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

# Graph Neural Network, GNN
class GNN(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(GNN, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel
        self.mlp1 = MLPs(2 * in_channel, 2 * in_channel)
        self.change1 = MLPs(2 * in_channel, in_channel)
        self.change2 = MLPs(2 * in_channel, in_channel)
        self.aff = AFF(in_channel, 4)
        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 3)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
            )
        if self.knn_num == 6:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel * 2, self.in_channel * 2, (1, 2)),
                nn.BatchNorm2d(self.in_channel * 2),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out_an = self.conv(out)
        out_an = self.change1(out_an)
        out_max = self.mlp1(out)
        out_max = out_max.max(dim=-1, keepdim=False)[0]
        out_max = out_max.unsqueeze(3)
        out_max = self.change2(out_max)
        out = self.aff(out_max, out_an)
        return out

class PointCN(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

# Context Position Attention, CPT
class CPT(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CPT, self).__init__()

        self.graph1_conv = nn.Conv2d(2, in_channels, kernel_size=1)
        self.graph2_conv = nn.Conv2d(2, in_channels, kernel_size=1)

        self.q = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.k = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.v = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.temperature = torch.sqrt(torch.tensor(in_channels))
        self.temperature2 = torch.sqrt(torch.tensor(in_channels))

    def forward(self, PointCN1, x):
        q = self.q(PointCN1).squeeze(3)
        k = self.k(PointCN1).squeeze(3)
        v = self.v(PointCN1).squeeze(3)

        # x = x.transpose(1, 3).contiguous()
        graph_1_coordinates = x[:, :2, :, :]  # 形状: [B, 1, N, 2]
        graph_2_coordinates = x[:, 2:4, :, :] # 形状: [B, 1, N, 2]
        graph_1 = self.graph1_conv(graph_1_coordinates)
        graph_2 = self.graph2_conv(graph_2_coordinates)
        graph_context = graph_1 + graph_2
        graph_context = graph_context.squeeze(3)

        graph_context_position = torch.matmul(q / self.temperature2, graph_context.transpose(1, 2))
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        attn = attn + graph_context_position
        # attn = self.dropout(F.softmax(attn, dim=-1))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v).unsqueeze(3)
        return output

# Multi-Branch Feed Forward Network, MBFFN
class MBFFN(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(MBFFN, self).__init__()
        inter_channels = int(in_channels // reduction)

        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.global_att_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels, inter_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.GELU(),
            nn.Conv2d(inter_channels, in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.sigmoid = nn.Sigmoid()

        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1)



    def forward(self, x):
        input_conv = self.conv_in(x)

        scale1 = self.local_att(input_conv)
        scale2 = self.global_att(input_conv)
        scale3 = self.global_att_max(input_conv)

        scale_out = scale1 + scale2 + scale3
        scale_out = self.sigmoid(scale_out)
        output_conv = self.conv_out(scale_out)
        out = output_conv + input_conv
        out = out + x
        return out

class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            trans(1, 2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(points),
            nn.ReLU(),
            nn.Conv2d(points, points, kernel_size=1)
        )
        self.conv3 = nn.Sequential(
            trans(1, 2),
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1, 2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_geo, x_down):
        embed = self.conv(x_geo)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class DP_OA_DOP_block(nn.Module):
    def __init__(self, channels, l2_nums, num):
        nn.Module.__init__(self)
        self.down1 = diff_pool(channels, l2_nums)
        self.l2 = []
        for _ in range(num):
            self.l2.append(OAFilter(channels, l2_nums))
        self.up1 = diff_unpool(channels, l2_nums)
        self.l2 = nn.Sequential(*self.l2)

    def forward(self, pre):
        x_down = self.down1(pre)
        x2 = self.l2(x_down)
        x_up = self.up1(pre, x2)
        return x_up

# Contextual Geometric Attention Module, CGA Module
class CGA_Module(nn.Module):
    def __init__(self, channels):
        super(CGA_Module, self).__init__()
        self.CPA = CPT(channels, channels)
        self.LayerNorm1 = nn.LayerNorm(channels, eps=1e-6)
        self.MBFFN = MBFFN(channels, channels)
        self.LayerNorm2 = nn.LayerNorm(channels, eps=1e-6)


    def forward(self, feature, Position_feature):
        # CPT
        CPT_feature = self.CPA(feature, Position_feature)
        CPT_feature = CPT_feature + feature
        # LayerNorm 1
        CPT_feature_LN1 = CPT_feature.squeeze(3).transpose(-1, -2)
        CPT_feature_LN1 = self.LayerNorm1(CPT_feature_LN1)
        CPT_feature_LN1 = CPT_feature_LN1.transpose(-1, -2).unsqueeze(3)
        # MBFFN
        MBFFN_feature = self.MBFFN(CPT_feature_LN1)
        # LayerNorm 2
        MBFFN_feature_LN2 = MBFFN_feature.squeeze(3).transpose(-1, -2)
        MBFFN_feature_LN2 = self.LayerNorm2(MBFFN_feature_LN2)
        MBFFN_feature_LN2 = MBFFN_feature_LN2.transpose(-1, -2).unsqueeze(3)
        # out
        out = CPT_feature_LN1 + MBFFN_feature_LN2

        return out



class sub_MGCANet(nn.Module):
    def __init__(self, i, net_channels, input_channel, depth, clusters, isInit=False, knn_num=9):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        print('channels:' + str(channels) + ', layer_num:' + str(self.layer_num))
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)
        if not isInit:
            self.conv641 = nn.Conv2d(channels, channels//2, kernel_size=1)
            self.conv642 = nn.Conv2d(channels, channels//2, kernel_size=1)
            self.conv643 = nn.Conv2d(channels, channels//2, kernel_size=1)

            self.MLP = MLPs(channels, channels)
            self.PointCN_256_128 = PointCN(256, 128)

            if i == 1:
                self.conv644 = nn.Conv2d(channels, channels//2, kernel_size=1)
                self.init_se = SE(channels//2)
                self.annular_convolution = nn.Sequential(
                    nn.Conv2d(channels, channels, (1, 3), stride=(1, 3)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, (1, 3)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, (1, 2)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            else:
                self.annular_convolution = nn.Sequential(
                    nn.Conv2d(channels, channels, (1, 3), stride=(1, 3)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, (1, 3)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )


        self.l1_1_1 = PointCN(channels)
        self.CGA1 = CGA_Module(channels)

        self.l1_1_2 = PointCN(channels)
        self.l1_1_3 = PointCN(channels)

        self.GNN = GNN(knn_num, channels)
        self.geo = DP_OA_DOP_block(channels, clusters, 3)



        self.l1_2_1 = PointCN(2 * channels, channels)
        self.l1_2_2 = PointCN(channels)

        self.CGA2 = CGA_Module(channels)
        self.l1_2_3 = PointCN(channels)


        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        self.linear1 = nn.Conv2d(channels, 2, kernel_size=1)


    def forward(self, data, xs, i, x_last=None, x_last2=None, init_out=None):
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)

        # CSMGC_Module
        if x_last is not None:
            # last stage feature
            x1_1_64 = self.conv643(x1_1)
            x_last_64 = self.conv641(x_last)
            x_last2_64 = self.conv642(x_last2)

            # GNN
            x1_1_64_graph = get_graph_feature(x1_1_64, k=3)
            x_last_64_graph = get_graph_feature(x_last_64, k=3)
            x_last2_64_graph = get_graph_feature(x_last2_64, k=3)

            if init_out is not None and i == 1:
                init_out_64 = self.conv644(init_out)
                init_out_64 = self.init_se(init_out_64)
                init_out_64_graph = get_graph_feature(init_out_64, k=3)
                combine_graph = torch.cat([x1_1_64_graph, x_last_64_graph, x_last2_64_graph, init_out_64_graph], dim=-1)
            else:
                # 对x1_1_64_graph, x_last_64_graph, x_last2_64_graph进行拼接
                combine_graph = torch.cat([x1_1_64_graph, x_last_64_graph, x_last2_64_graph], dim=-1)

            # ANN + MLP
            dgcnn = self.annular_convolution(combine_graph)
            dgcnn = self.MLP(dgcnn)
            x1_1 = torch.cat([x1_1, dgcnn], dim=1)
            x1_1 = self.PointCN_256_128(x1_1)
        x1_1_1 = self.l1_1_1(x1_1)

        # GCA + PointCN
        x1_1_1 = self.CGA1(x1_1_1, data)
        x1_1_2 = self.l1_1_2(x1_1_1)
        x1_1_3 = self.l1_1_3(x1_1_2)
        x1_1 = x1_1_3

        # GNN
        x1_1 = self.GNN(x1_1)

        # DP_OA_DOP_block OANet
        x2 = self.geo(x1_1)
        x2_out = torch.cat([x1_1, x2], dim=1)

        # POintCN + CGA
        x1_2_1 = self.l1_2_1(x2_out)
        x1_2_2 = self.l1_2_2(x1_2_1)
        x1_2_2 = self.CGA2(x1_2_2, data)
        x1_2_3 = self.l1_2_3(x1_2_2)
        out = x1_2_3


        logits = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
        logits1, indices = torch.sort(logits, dim=-1, descending=True)

        x_out, feature_out = down_sampling(xs, indices, out)
        w = self.linear1(feature_out)
        e_hat = weighted_8points(x_out, w)
        x1, x2 = xs[:, 0, :, :2], xs[:, 0, :, 2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(
            batch_size, 1, num_pts, 1)

        return logits, e_hat, residual, out, x1_1

# Cross-Stage Multi-Graph Consensus Module, CSMGC
class CSMGC(nn.Module):
    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)
        self.annular_convolution = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels * 2, (1, 3), stride=(1, 3)),
                nn.BatchNorm2d(in_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels * 2, in_channels * 2, (1, 2)),
                nn.BatchNorm2d(in_channels * 2),
                nn.ReLU(inplace=True),
            )
        self.MLP = MLPs(in_channels *2 , out_channels)


    def forward(self, Stage_1, Stage_2, Stage_3):
        S1_graph = get_graph_feature(Stage_1, k=2)
        S2_graph = get_graph_feature(Stage_2, k=2)
        S3_graph = get_graph_feature(Stage_3, k=2)
        Combine_Stage_Graph = torch.cat([S1_graph, S2_graph, S3_graph], dim=-1)
        ANN_out = self.MLP(self.annular_convolution(Combine_Stage_Graph))
        out = ANN_out + Stage_3
        return out



class MGCANet(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = 2
        depth_each_stage = config.net_depth // (config.iter_num + 1)
        self.subnetwork_init = sub_MGCANet(
            0,config.net_channels, 4, depth_each_stage, config.clusters, isInit=True, knn_num=6)
        self.subnetwork = [sub_MGCANet(i_num, config.net_channels, 6, depth_each_stage,
                                     config.clusters, isInit=False, knn_num=6) for i_num in range(self.iter_num)]
        self.subnetwork = nn.Sequential(*self.subnetwork)

        self.CSMGC = CSMGC(config.net_channels, config.net_channels)
        self.M1 = MBSE(config.net_channels,config.net_channels // 2, 1)
        self.M2 = MBSE(config.net_channels,config.net_channels // 2, 1)

        self.covn = nn.Conv2d(128, 1, kernel_size=1)
        self.linear1 = nn.Conv2d(128, 2, kernel_size=1)



    def forward(self, data):
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]

        input = data['xs'].transpose(1, 3)
        res_weights, res_e_hat = [], []
        stage_out = []

        logits, e_hat, residual, out, x_last = self.subnetwork_init(
            input, data['xs'], 0)
        stage_out.append(out)
        init_out = out
        More_weight = out
        res_weights.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual, out, x_last = self.subnetwork[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()],
                          dim=1), data['xs'], i,x_last, out, init_out)
            More_weight = torch.cat([More_weight, out], dim=1)
            res_weights.append(logits), res_e_hat.append(e_hat)
            stage_out.append(out)

        sub_l_input = self.CSMGC(stage_out[0], stage_out[1], stage_out[2])
        More_weight = self.M2(self.M1(sub_l_input))
        feature_out = self.covn(More_weight)
        logits = torch.squeeze(torch.squeeze(feature_out, 3), 1)
        logits1, indices = torch.sort(logits, dim=-1, descending=True)
        x_out, feature_out = down_sampling(data['xs'], indices, More_weight)
        w = self.linear1(feature_out)
        e_hat = weighted_8points(x_out, w)
        res_weights.append(logits), res_e_hat.append(e_hat)
        return res_weights, res_e_hat

def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='L')
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv

def weighted_8points(x_in, logits):
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    # weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)

    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


