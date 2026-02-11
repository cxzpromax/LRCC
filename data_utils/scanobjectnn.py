import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset, dataset
import torch.utils.data as data
import torch


class ScanObjectNN(Dataset):
    def __init__(self,data_path,split='train',num_points=1024):
        self.split = split
        self.BASE_DIR=data_path
        self.data, self.label = self.load_scanobjectnn_data()
        self.num_points = num_points
               

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
        return all_data, all_label

    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points,:].astype(np.float)
        label = self.label[item]
        # if self.split == 'train':
        #     pointcloud = self.translate_pointcloud(pointcloud)
        #     np.random.shuffle(pointcloud)

        pointcloud=torch.FloatTensor(pointcloud)
        label=torch.LongTensor([label])
        return pointcloud, label


"""Obtain training_loader and test_loader"""
def get_sets(data_path, batch_size):
    train_data=ScanObjectNN(data_path,split='train')
    train_loader=data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=2)

    test_data=ScanObjectNN(data_path,split='test')
    test_loader=data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,num_workers=2)
    
    return train_loader, test_loader



if __name__=='__main__':
    data_path='/media/one/系统/chenzhuang/Datasets/h5_files/main_split_nobg'
    dataset=ScanObjectNN(data_path, split='train')
    print(dataset.data[0].shape)  # (2048, 3)
    print(dataset.data[1].shape)  # (2048, 3)
    # np.savetxt('demo1.txt', dataset.data[0])
    # np.savetxt('demo2.txt', dataset.data[1])
    # np.savetxt('demo3.txt', dataset.data[2])
    # np.savetxt('demo4.txt', dataset.data[3])
    #
    # np.savetxt('demo5.txt', dataset.data[4])
    # np.savetxt('demo6.txt', dataset.data[5])
    # np.savetxt('demo7.txt', dataset.data[6])
    # np.savetxt('demo8.txt', dataset.data[7])
    #
    # np.savetxt('demo9.txt', dataset.data[8])
    # np.savetxt('demo10.txt', dataset.data[9])
    # np.savetxt('demo11.txt', dataset.data[10])
    # np.savetxt('demo12.txt', dataset.data[11])






