import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')


'''
pc:point cloud；
normalize the point cloud
'''
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)  # axis=0, order by column
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    the farthest point sample(FPS) method
    Input:
        point: all point cloud of an item, [N, D](N lines, D dimension)
        ndarray type
        npoint: number of samples
    Return:
        centroids: sampled point cloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]  # first three columns save location information
    centroids = np.zeros((npoint,))  # init sampled index
    distance = np.ones((N,)) * 1e10  # save distance between all points and sampled point set
    farthest = np.random.randint(0, N)  # select the first point randomly
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)  # -1代表在最后一个维度操作
        mask = dist < distance
        distance[mask] = dist[mask]  # 更新最小值
        farthest = np.argmax(distance, -1)
    sample_points = point[centroids.astype(np.int32)]
    return sample_points

class ModelNetDataLoader(Dataset):
    """
    Input:
        root: the dataset root dir(such as ../data/modelnet40_normal_resampled)
        npoint: number of sampled points, default=1024
        split: train data or test data default=train
        uniform: methods to sample points, default method: false:select the top-npoint points;
        true:use the farthest point sample method
        normal_channel(bool): if the input data has 3 channel, normal_channel
        should be True, else false
    Return:
        centroids: sampled point cloud index, [npoint, D]
    """
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')  # obtain all categories
        # save categories as list
        self.cat = [line.rstrip() for line in open(self.catfile)]
        # 封装成字典，如{'airplane':0, 'bathtub':1,...}
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}  # 分别保存train和test物品的id
        # 'train':['airplane_0001','airplane_0002',...]
        # 'test':['airplane_0101','airplane_0102',...]
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert (split == 'train' or split == 'test')
        # x.split('_')[0:-1]，得到每个样本的类别名称，去掉_和后面的编号
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple(label_name, item_path)
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        self._get_item()

    def __len__(self):
        return len(self.datapath)

    def _get_item(self):
        for index in range(len(self.datapath)):
            # print('re_calculate')
            fn = self.datapath[index]  # (fn[0]:class_name, fn[1]:path,....txt)
            cls = self.classes[fn[0]]  # (cls:class_id)
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',', usecols=(0, 1, 2)).astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]  # select the top points
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # 1024 * 3, numpy
            self.cache[index] = (point_set, cls)

    def __getitem__(self, index):
        return self.cache[index]


class ModelNetData_10_Loader(Dataset):
    """
    Input:
        root: the dataset root dir(such as ../data/modelnet40_normal_resampled)
        npoint: number of sampled points, default=1024
        split: train data or test data default=train
        uniform: methods to sample points, default method: false:select the top-npoint points;
        true:use the farthest point sample method
        normal_channel(bool): if the input data has 3 channel, normal_channel
        should be True, else false
    Return:
        centroids: sampled point cloud index, [npoint, D]
    """
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')  # obtain all categories
        # save categories as list
        self.cat = [line.rstrip() for line in open(self.catfile)]
        # 封装成字典，如{'airplane':0, 'bathtub':1,...}
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}  # 分别保存train和test物品的id
        # 'train':['airplane_0001','airplane_0002',...]
        # 'test':['airplane_0101','airplane_0102',...]
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        assert (split == 'train' or split == 'test')
        # x.split('_')[0:-1]，得到每个样本的类别名称，去掉_和后面的编号
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple(label_name, item_path)
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split, len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        self._get_item()

    def __len__(self):
        return len(self.datapath)

    def _get_item(self):
        for index in range(len(self.datapath)):
            # print('re_calculate')
            fn = self.datapath[index]  # (fn[0]:class_name, fn[1]:path,....txt)
            cls = self.classes[fn[0]]  # (cls:class_id)
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',', usecols=(0, 1, 2)).astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]  # select the top points
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # 1024 * 3, numpy
            self.cache[index] = (point_set, cls)

    def __getitem__(self, index):
        return self.cache[index]
