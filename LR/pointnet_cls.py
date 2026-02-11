import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet import PointNetEncoder, feature_transform_reguliarzer


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=False, channel=channel)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x, trans, trans_feat = self.feat(x)  # 两个变换矩阵被取消了
        x, features = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # print(x.shape)  # batch_size * k (8 * 40)
        x = F.log_softmax(x, dim=1)  # 按行进行softmax操作
        return x, features

class get_loss(torch.nn.Module):
    def __init__(self, weight=0.001):
        super(get_loss, self).__init__()
        self.weight = weight

    def get_legal_id(self, used_index_list, data_index, num_point):
        mask = torch.zeros(num_point)
        used_index = used_index_list[data_index]
        mask[used_index] = 1
        legal_index = torch.where(mask == 0)[0]
        return legal_index

    def feature_refinement(self, point_feat, refine_times):
        device = point_feat.device
        num_point = point_feat.shape[2]
        batch_size = point_feat.shape[0]

        feat_list = []
        used_index_list = [torch.LongTensor([]).to(device) for _ in range(batch_size)]

        # point_feat： B * F * P
        for i in range(refine_times):
            hie_feat_list = []
            for data_index, single_data in enumerate(point_feat):
                legal_index = self.get_legal_id(used_index_list, data_index, num_point)
                legal_feat = single_data[:, legal_index]

                max_feat, max_index = torch.max(legal_feat, -1)
                max_index = torch.unique(max_index).detach()
                hie_feat_list.append(max_feat)
                used_index_list[data_index] = torch.cat((used_index_list[data_index], max_index))

            hie_feat_list = torch.stack(hie_feat_list, 0)
            feat_list.append(hie_feat_list)

        feat_list = torch.stack(feat_list, 0)
        # feat_list=feat_list.permute(1,0,2)

        return feat_list

    def forward(self, pred, target, features, refine_times):
        loss = F.nll_loss(pred, target)

        """obtain the high-order features"""
        feature_list = self.feature_refinement(features, refine_times).permute(1, 2, 0)  # torch.Size([8, 1024, 5])
        """1. showing the nuclear norm for different samples"""
        data = []
        # for i in range(feature_list.shape[0]):
        #     norm = torch.linalg.norm(feature_list[i, :, :], ord='nuc')
        #     data.append(norm.detach().cpu().numpy())
        # np.savetxt('data.txt', data)
        # exit()

        # """2. showing the rank value for different samples"""
        # for i in range(feature_list.shape[0]):
        #     norm = torch.matrix_rank(feature_list[i, :, :])
        #     # norm = torch.linalg.norm(feature_list[i, :, :], ord='nuc')
        #     data.append(norm.detach().cpu().numpy())
        # np.savetxt('data_nuclear.txt', data)
        # exit()



        # """1. showing the nuclear norm for different samples"""
        # data = []
        # # for i in range(feature_list.shape[0]):
        # #     norm = torch.linalg.norm(feature_list[i, :, :], ord='nuc')
        # #     data.append(norm.detach().cpu().numpy())
        # # np.savetxt('data.txt', data)
        # # exit()
        #
        # # """2. showing the rank value for different samples"""
        # for i in range(feature_list.shape[0]):
        #     norm = torch.matrix_rank(feature_list[i, :, :])
        #     # norm = torch.linalg.norm(feature_list[i, :, :], ord='nuc')
        #     data.append(norm.detach().cpu().numpy())
        # np.savetxt('data_nuclear.txt', data)
        # exit()

        # calculate the nuclear norm for each batch
        nuclear_loss = torch.tensor(0.).cuda()
        for i in range(feature_list.shape[0]):
            nuclear_loss = nuclear_loss + torch.linalg.norm(feature_list[i, :, :], ord='nuc')
        nuclear_loss = nuclear_loss / feature_list.shape[0]
        # print(loss, nuclear_loss * self.weight)
        # exit()
        """low rank constraints"""
        return loss + nuclear_loss * self.weight






