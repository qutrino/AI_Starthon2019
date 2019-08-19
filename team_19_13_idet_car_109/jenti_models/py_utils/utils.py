import torch
import torch.nn as nn

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _decode_one(tl_heat, br_heat, tl_off, br_off):

    #print("tloff", tl_off)
    #print("br_off", br_off)

    #print(tl_off.size())

    batch, cat, height, width = tl_heat.size()
    #print(batch, cat, height, width)

    tl_heat = tl_heat.view(batch, -1)
    br_heat = br_heat.view(batch, -1)

    tl_max_ind = torch.argmax(tl_heat,dim=1)
    br_max_ind = torch.argmax(br_heat,dim=1)



    tl_off = tl_off.view(batch,2, -1)
    br_off = br_off.view(batch,2, -1)

    #print("br_off:", br_off.shape)

    tl_x_off = torch.diagonal(tl_off[:,0, tl_max_ind], offset=0, dim1=0, dim2=-1)
    tl_y_off = torch.diagonal(tl_off[:,1, tl_max_ind], offset=0,  dim1=0, dim2=-1)

    br_x_off = torch.diagonal(br_off[:,0, br_max_ind], offset=0,  dim1=0, dim2=-1)
    br_y_off = torch.diagonal(br_off[:,1, br_max_ind], offset=0,  dim1=0, dim2=-1)


    #print("BRX:", br_x_off.shape)
    #print("tl_max_ind:", tl_max_ind.shape)


    tl_y  = (tl_max_ind / width).int().float() + tl_y_off
    tl_x  = (tl_max_ind % width).int().float() + tl_x_off

    br_y  = (br_max_ind / width).int().float() + br_y_off
    br_x  = (br_max_ind % width).int().float() + br_x_off

    return torch.stack([tl_x / width, tl_y / height, (br_x-tl_x)/width, (br_y-tl_y)/height ], dim=1 )


class upsample(nn.Module):
    def __init__(self, scale_factor):
        super(upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor)

class merge(nn.Module):
    def forward(self, x, y):
        return x + y

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1):
        super(residual, self).__init__()
        p = (k - 1) // 2

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(p, p), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (k, k), padding=(p, p), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

class corner_pool(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(corner_pool, self).__init__()
        self._init_layers(dim, pool1, pool2)

    def _init_layers(self, dim, pool1, pool2):
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2
