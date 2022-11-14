import numpy as np
import torch

from base.dataset import DatasetManager
from configs import *

#use BatchManager to get train/val/test data,num_user,num_item,mean and std of ratings
#in train
class BatchManager:
    def __init__(self, kind):
        self.kind = kind
        dataset_manager = DatasetManager(kind, N_SHOT)
        self.train_data = self._torch_version(
                          dataset_manager.get_train_data())
        self.valid_data = self._torch_version(
                          dataset_manager.get_valid_data())
        self.test_data = self._torch_version(
                         dataset_manager.get_test_data())

        self.n_user = dataset_manager.get_n_user()
        self.n_item = dataset_manager.get_n_item()
        self.mu = torch.mean(torch.as_tensor(self.train_data[:, 2],
                                             dtype=torch.float32))
        self.std = torch.std(torch.as_tensor(self.train_data[:, 2],
                                             dtype=torch.float32))
    def _torch_version(self,data):
        data = torch.as_tensor(torch.from_numpy(data),dtype = torch.long)
        return data
