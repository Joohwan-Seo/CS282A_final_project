import os
import numpy as np
import random
import time
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
import pickle

#################### Read file ####################
def read_off(path):
    assert os.path.exists(path)
    file = open(path, 'r')

    # line 0
    assert file.readline().strip() == "OFF"
    # line 1: no edges
    nv, nf, _ = file.readline().strip().split(" ")

    # other lines
    v, f = [], []
    
    for i in range(int(nv)):
        vals = file.readline().strip().split(" ")
        v.append([float(val) for val in vals])
    
    for i in range(int(nf)):
        vals = file.readline().strip().split(" ")[1:]
        f.append([int(val) for val in vals])

    return v, f


#################### For transformer ####################
class Normalize(object):
    def __call__(self, data_in):
        assert data_in.ndim==2
        # column-wise: mean 0
        data_out = data_in - np.mean(data_in, axis=0)
        # use the row with the largest L2 norm
        data_out = data_out / np.max(np.linalg.norm(data_out, axis=1))

        return  data_out

class RotateXYZ(object):
    def __call__(self, data_in):
        assert data_in.ndim==2

        theta_dict = {axis: 2 * np.pi * random.random() for axis in ["x", "y", "Z"]}
        
        for axis in theta_dict.keys():    
            theta = theta_dict[axis]
            if axis == "x":
                R = np.array([
                        [1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]
                    ])
            elif axis =="y":
                R = np.array([
                        [np.cos(theta), 0, -np.sin(theta)],
                        [0, 1, 0],
                        [np.sin(theta), 0, np.cos(theta)]
                    ])
            elif axis == "z":
                R = np.array([
                        [np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]
                    ])
        
            data_in = np.matmul(data_in, R.T)

        return  data_in
    
    
class AddGaussianNoise(object):
    def __init__(self, var = 0.01):
        self.var = var

    def __call__(self, data_in):
        assert data_in.ndim==2
        data_out = data_in + np.random.normal(0, self.var, data_in.shape)
        return  data_out

def test_transform():
    return transforms.Compose([Normalize(), 
                               transforms.ToTensor() 
                              ])

#################### Custom dataset ####################
class PointCloudData(Dataset):
    def __init__(self, data_split:str, method:str, task="Classification",
                 acc_type = 'valid', transform = None,
                 data_folder = 'Dataset'):

        assert task in ['Classification', 'Segmentation']
        assert data_split in ['train', 'test']
        assert method in ['random', 'farthest']
        assert acc_type in ['valid', 'test']

        # get file
        self.task = task
        self.file_name = f"{data_folder}/{task}_{data_split}_{method}.pkl"
        dataset = open(self.file_name, 'rb')
        self.files = pickle.load(dataset)
        dataset.close()

        # assign transform
        self.acc_type = acc_type
        if self.acc_type == 'valid':
            self.transform = transform
        elif self.acc_type == 'test':
            self.transform = test_transform()
        
        # classes
        if self.task == 'Classification':
            class_names = ["bathtub", "bed", "chair", "desk", "dresser",
                        "monitor", "night_stand", "sofa", "table", "toilet"]
            self.classes = {cat: i for i, cat in enumerate(class_names)}
        
    def __len__(self):
        return len(self.files)

    def do_transform(self, v):
        return self.transform(v)

    def __getitem__(self, idx):
        cat = self.files[idx]['category']
        pointcloud = self.files[idx]['sampled_points']
        pointcloud = self.do_transform(pointcloud)

        if self.task == "Classification":    
            return {'pointcloud': pointcloud.squeeze(),
                    'category': self.classes[cat]}

        elif self.task == "Segmentation":
            return {'pointcloud': pointcloud.squeeze(), 
                    'category': cat.astype(int)}
