import numpy as np

from base.dataset import DatasetManager
from configs import *

#use BatchManager to get train/val/test data,num_user,num_item,mean and std of ratings
#in train
class BatchManager:
    def __init__(self, kind):
        self.kind = kind
        dataset_manager = DatasetManager(kind, N_SHOT)
        self.train_data = np.concatenate(
            [
                dataset_manager.get_train_data(),
                dataset_manager.get_valid_data()
            ],
            axis=0)
        self.test_data = dataset_manager.get_test_data()

        self.n_user = dataset_manager.get_n_user()
        self.n_item = dataset_manager.get_n_item()
        self.mu = np.mean(self.train_data[:, 2])
        self.std = np.std(self.train_data[:, 2])
