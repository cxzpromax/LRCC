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

        self.fc0 = nn.Linear(2048, 1024)
        self.bn0 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, curvatures):
        # x, trans, trans_feat = self.feat(x)  # 两个变换矩阵被取消了
        x = self.feat(x, curvatures)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # print(x.shape)  # batch_size * k (8 * 40)
        x = F.log_softmax(x, dim=1)  # 按行进行softmax操作
        return x

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        return loss



if __name__ == '__main__':
    point_net_cls = get_model(40, normal_channel=False)
    input = torch.randn(16, 3, 1024)
    output = point_net_cls(input)
    print(output)

    # classifier(points, to_categorical(label, num_classes))





