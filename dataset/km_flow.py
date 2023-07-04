import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class KMFlowDataset:
    def __init__(self, data_dir, batch_size=128, load_subset=False):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.load_subset = load_subset
        
        self.ground_truth = np.load(os.path.join(data_dir, 'km_flow.npy'))
        with np.load(os.path.join(data_dir, 'km_flow_sampled.npz'), allow_pickle=True) as f:
            self.sampled_data = f['u3232'].copy().astype(np.float32)
            self.idx_lst = f['idx_lst'].copy()
        
        if load_subset:
            train_X = self.sampled_data[:4]
            train_y = self.ground_truth[:4]
        else:
            train_X = self.sampled_data[:32]
            train_y = self.ground_truth[:32]
        valid_X = self.sampled_data[32:36]
        valid_y = self.ground_truth[32:36]
        test_X = self.sampled_data[36:]
        test_y = self.ground_truth[36:]
        
        # stats_path = os.path.join(data_dir, 'stats.npz')
        # if os.path.exists(stats_path):
        #     self.stats = np.load(stats_path)
        # else:
        #     self.stats = {}
        #     self.process_stats()
        
        self.train_data = KMFlowBase(train_X, train_y)
        self.valid_data = KMFlowBase(valid_X, valid_y)
        self.test_data = KMFlowBase(test_X, test_y)

    @property
    def train_loader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    @property
    def valid_loader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)
    
    @property
    def test_loader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
    
    def process_stats(self):
        self.stat['mean'] = np.mean(self.all_data[self.train_idx_lst[:]].reshape(-1, 1))
        self.stat['scale'] = np.std(self.all_data[self.train_idx_lst[:]].reshape(-1, 1))
        data_mean = self.stat['mean']
        data_scale = self.stat['scale']


class KMFlowBase(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X.reshape(-1, 1, 256, 256)
        self.y = y.reshape(-1, 1, 256, 256)
        
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32)
