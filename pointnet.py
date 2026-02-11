import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class STN3d(nn.Module):  # T_Net 3*3
    def __init__(self, channel=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):  # T-Net 64*64
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


'''
feature_transform = True, use T_net64 to transform the learned features
global_feat=True返回全局特征；global_feat=False返回全局特征+局部特征
x: batch_size * channel * size (batch_size * 3 * 1024)
'''
class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        # self.stn = STN3d(channel)  T-Net网络不加上其实对结果影响不大
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:  # 是否启动T-Net 64*64
            self.fstn = STNkd(k=64)

    def uniform_to_scale(self, seq_a, seq_b):
        min_b, max_b = torch.min(seq_b), torch.max(seq_b)
        scaled_a = (seq_a - torch.min(seq_a)) / (torch.max(seq_a) - torch.min(seq_a)) * (max_b - min_b) + min_b
        return scaled_a

    def forward(self, x, curvatures):
        B, D, N = x.size()  # B:Batch size, D: dimension, channel(3), N: 1024 points
        # trans = self.stn
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # 8 * 1024 * 1024 (batch_size, channel, npoints)
        # x = torch.einsum('ijk,ik->ijk', x, curvatures)  # torch.Size([8, 1024, 1024])
        x = torch.max(x, 2, keepdim=True)[0]  # 8 * 1024

        x = x.view(-1, 1024)

        # normalization and Unified scale
        # curvatures = self.uniform_to_scale(curvatures, x)
        curvatures, _ = torch.sort(curvatures, descending=True)  # torch.Size([8, 2048])

        # feature fusion
        x = torch.cat((x, curvatures), dim=1)  # torch.Size([8, 2048])

        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], dim=1)  # place on tht right

'''
文件的最后还定义了一个函数：feature_transform_reguliarzer，
实现前面在网络流程部分提到的对特征空间进行对齐时，需要添加正则约束，
让求解出来的矩阵接近于正交矩阵，也就是实现这个式子，
A就是估计的k*k矩阵（函数输入）。模块返回此正则项，后面在计算loss的时候会用到。
'''

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


# stn_3d = STN3d()
# print(stn_3d)

# data = torch.randn((64, 3, 1024))  # bitch_size, channel, size
# model = PointNetEncoder(global_feat=True, feature_transform=False)
# output = model(data)
# print(output[0], output[0].shape)  # 64*1024

