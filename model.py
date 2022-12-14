import torch
import numpy as np
from torch.nn import functional as F
from configs import *
from base.dataset import DatasetManager

class PretrianModel(torch.nn.Module):
    def __init__(self,batch_manager):
        super(PretrianModel,self).__init__()
        self.n_row = batch_manager.n_user
        self.n_col = batch_manager.n_item
        self.mu = batch_manager.mu
        self.std = batch_manager.std
        self._u_feat = self._create_p_or_q_variable(self.n_row,
                                                    PRE_RANK,
                                                    self.mu,
                                                    self.std).requires_grad_(True)
        self._i_feat = self._create_p_or_q_variable(self.n_col,
                                                    PRE_RANK,
                                                    self.mu,
                                                    self.std).requires_grad_(True)

    def _create_p_or_q_variable(self,n, rank, mu, std):
        # 按特定分布初始化p,q
        _mu = np.sqrt(mu / rank)
        _std = np.sqrt((np.sqrt(mu * mu + std * std) - mu) / rank)
        mat = torch.empty(n,rank)
        return torch.as_tensor(torch.nn.init.trunc_normal_(mat,
                                                           _mu,
                                                           _std,
                                                           _mu-2*_std,
                                                           _mu+2*_std))

    def get_u_feat(self):
        return self._u_feat
    def get_i_feat(self):
        return self._i_feat

    def forward(self, u, i, r):
        u_lookup = F.embedding(u,self._u_feat)
        i_lookup = F.embedding(i,self._i_feat)
        r_hat = torch.sum(torch.multiply(u_lookup,i_lookup),1)

        MSE = torch.sum(torch.square(r_hat - r))

        rmse = torch.sqrt(torch.mean(torch.square(r_hat - r)))

        reg_loss = torch.add(torch.sum(torch.square(self._u_feat))
                            , torch.sum(torch.square(self._i_feat)))
        loss = MSE + PRE_LAMBDA*reg_loss
        return loss,rmse
def pre_test(model,batchmanager):
    test_data = torch.from_numpy(batchmanager.test_data)
    u = torch.as_tensor(test_data[:, 0], dtype=torch.long)
    i = torch.as_tensor(test_data[:, 1], dtype=torch.long)
    r = torch.as_tensor(test_data[:, 2], dtype=torch.float32)
    model.eval()
    test_loss, rmse = model(u, i, r)
    print("test loss:{} RMSE:{}".format(test_loss, rmse))

class BaseModel(torch.nn.Module):
    def __init__(self, batch_manager):
        super(BaseModel, self).__init__()
        self.n_row = batch_manager.n_user
        self.n_col = batch_manager.n_item

    def set_feats(self,feat_u,feat_i):
        self._feat_u = torch.from_numpy(feat_u).requires_grad_(True)
        self._feat_i = torch.from_numpy(feat_i).requires_grad_(True)

    def get_feats(self):
        return self._feat_u, self._feat_i

    def _inner_pro(self, u, i):
        u_lookup = F.embedding(u, self._feat_u)
        i_lookup = F.embedding(i, self._feat_i)
        r_hat = torch.sum(torch.multiply(u_lookup, i_lookup), 1)
        return r_hat

    def mse_loss(self, u, i, r, k):
        k = torch.as_tensor(k)
        r_hat = self._inner_pro(u, i)

        SSE = torch.sum(torch.square(r_hat - r))
        reg_loss = torch.add(torch.sum(torch.square(self._feat_u))
                             , torch.sum(torch.square(self._feat_i)))
        local_loss = torch.sum(torch.square(r_hat - r) * k) + LOCAL_LAMBDA * reg_loss

        return SSE, local_loss

    def forward(self, u, i):
        r_hat = self._inner_pro(u,i)

        return r_hat

