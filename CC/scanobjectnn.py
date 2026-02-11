import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset, dataset
import torch.utils.data as data
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class ScanObjectNN(Dataset):
    def __init__(self,data_path,split='train',num_points=1024):
        self.split = split
        self.BASE_DIR=data_path

        self.num_points = num_points
        self.cache = {}

        self.load_scanobjectnn_data()

    def translate_pointcloud(self,pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud
    
    
    def load_scanobjectnn_data(self):
        # self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # self.BASE_DIR='D:/Computer_vision/Dataset/ScanObjectNN/h5_files/h5_files/main_split'
        # DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        all_data = []
        all_label = []

        if self.split=='train':
            partition='training'
        else:
            partition='test'

        # h5_name = self.BASE_DIR + '/data/' + partition + '_objectdataset_augmentedrot_scale75.h5'
        h5_name = os.path.join(self.BASE_DIR, partition + '_objectdataset.h5')
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        # print(all_data.shape)  # 2039 * 2048 * 3

        # sample 1024 points
        print('Estimate the curvature for each point')
        for i in tqdm(range(all_data.shape[0])):
            point_set = all_data[i][:self.num_points,:].astype(np.float)  # 1024 * 3
            label = all_label[i]
            curvatures = estimate_single_curvature(point_set, k=10)

            point_set = torch.FloatTensor(point_set)
            label = torch.LongTensor([label])
            curvatures = torch.FloatTensor(curvatures)

            # print(point_set.shape, type(point_set))
            # print(label.shape, type(label))
            # print(curvatures.shape, type(curvatures))
            self.cache[i] = (point_set, label, curvatures)

    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, item):
        return self.cache[item]


def estimate_single_curvature(points, k=10):
    """
    Estimate the curvature of each point in a point cloud using PCA.

    Parameters:
    - points: An n x 3 NumPy array of 3D points.
    - k: Number of nearest neighbors to consider for each point.

    Returns:
    - A NumPy array of curvature estimates for each point.
    """
    # Initialize an empty array to store curvature estimates
    curvatures = np.zeros(len(points), dtype=np.float32)

    # Find the k-nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, indices = nbrs.kneighbors(points)

    for i in range(len(points)):
        # Get the indices of the k-nearest neighbors (excluding the point itself)
        neighbor_indices = indices[i, 1:]

        # Extract the coordinates of the neighbors
        neighbors = points[neighbor_indices]

        # Center the neighbors around the current point
        centered_neighbors = neighbors - points[i]

        # Compute the covariance matrix
        cov_matrix = np.cov(centered_neighbors.T)

        # Perform eigenvalue decomposition
        eigenvalues, _ = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Estimate the curvature using the smallest eigenvalue
        if np.sum(eigenvalues) > 0:
            curvature = eigenvalues[-1] / np.sum(eigenvalues)
        else:
            curvature = 0

        curvatures[i] = curvature
    return curvatures


"""Obtain training_loader and test_loader"""
def get_sets(data_path, batch_size):
    train_data=ScanObjectNN(data_path,split='train')
    train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=2)

    test_data=ScanObjectNN(data_path,split='test')
    test_loader=data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,num_workers=2)
    
    return train_loader, test_loader



if __name__=='__main__':
    data_path='/media/one/系统/chenzhuang/Datasets/h5_files/main_split_nobg'
    dataset=ScanObjectNN(data_path,split='train')
    
    picked_index=100
    picked_data=dataset[picked_index]
    
    a,b,c=get_sets(data_path,10,10)