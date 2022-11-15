import time
import math
import numpy as np
import torch
from configs import *
from batch import BatchManager
from model import BaseModel


def _create_p_or_q(n, rank, batch_manager):
    #initialize p,q according to specfific distribution
    mu = batch_manager.mu
    std = batch_manager.std

    _mu = math.sqrt(mu / rank)
    _std = math.sqrt((math.sqrt(mu * mu + std * std) - mu) / rank)
    return np.random.normal(_mu, _std, [n, rank])

class LocalModel:
    def __init__(self, model, anchor_idx, anchor_manager, batch_manager):
        self.model = model
        self.batch_manager = batch_manager
        self.anchor_idx = anchor_idx
        self.anchor_manager = anchor_manager
        self.feat_u = _create_p_or_q(batch_manager.n_user,
                                          LOCAL_RANK,
                                          batch_manager)
        self.feat_i = _create_p_or_q(batch_manager.n_item,
                                          LOCAL_RANK,
                                          batch_manager)
        self.train_k = self.anchor_manager.get_train_k(self.anchor_idx)
        self.test_k = self.anchor_manager.get_test_k(self.anchor_idx)
    def _update_r_hats(self):
        model = self.model
        batch_manager = self.batch_manager

        train_data = batch_manager.train_data
        train_r_hat = model(train_data[:,0],
                            train_data[:,1])
        test_data = batch_manager.test_data
        test_r_hat = model(test_data[:,0],
                           test_data[:,1])
        self.train_r_hat = train_r_hat
        self.test_r_hat = test_r_hat

    def train(self):
        model = self.model
        batch_manager = self.batch_manager
        anchor_idx = self.anchor_idx
        anchor_manager = self.anchor_manager
        init_feat_u = self.feat_u
        init_feat_i = self.feat_i
        train_k = self.train_k

        train_data = batch_manager.train_data
        prev_train_rmse = 5.0
        sum_batch_sse = 0.0
        n_batch = 0

        model.set_feats(init_feat_u,init_feat_i)

        #define optimizer
        opt = torch.optim.SGD([i for i in model.get_feats()],
                               lr=LOCAL_LEARNING_RATE,
                               momentum=0.9)

        for iter in range(1, 100 + 1):
            for m in range(0, train_data.shape[0], BATCH_SIZE):
                end_m = min(m + BATCH_SIZE, train_data.shape[0])
                u = train_data[m:end_m, 0]
                i = train_data[m:end_m, 1]
                r = torch.as_tensor(train_data[m:end_m, 2],dtype=torch.float32)
                k = train_k[m:end_m]

                opt.zero_grad()
                sse, loss = model.mse_loss(u, i, r, k)
                loss.backward()
                opt.step()

                sum_batch_sse += sse
                n_batch += u.shape[0]

            train_rmse = math.sqrt(sum_batch_sse / n_batch)
            train_rmse = math.sqrt(sum_batch_sse / n_batch)
            if iter % 10 == 0:
                print('  - ITER [{:3d}]'.format(iter), train_rmse)

            if abs(prev_train_rmse - train_rmse) < 1e-4:
                break
            prev_train_rmse = train_rmse
            feat_u, feat_i = model.get_feats()
            sum_batch_sse, n_batch = 0, 0
        self.feat_u, self.feat_i = feat_u, feat_i

        self._update_r_hats()

        self.train_k = train_k