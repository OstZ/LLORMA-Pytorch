import torch
import numpy as np
from torch.nn import functional as F
from configs import *
from base.dataset import DatasetManager

class pretrian_model(torch.nn.Module):
    def __init__(self,batch_manager):
        super(pretrian_model,self).__init__()
        self.n_row = batch_manager.n_user
        self.n_col = batch_manager.n_item
        self.mu = batch_manager.mu
        self.std = batch_manager.std
        self._u_feat = self._create_p_or_q_variable(self.n_row, PRE_RANK, self.mu, self.std).requires_grad_(True)
        self._i_feat = self._create_p_or_q_variable(self.n_col, PRE_RANK, self.mu, self.std).requires_grad_(True)

    def _create_p_or_q_variable(self,n, rank, mu, std):
        # 按特定分布初始化p,q
        _mu = np.sqrt(mu / rank)
        _std = np.sqrt((np.sqrt(mu * mu + std * std) - mu) / rank)
        mat = torch.empty(n,rank)
        return torch.as_tensor(torch.nn.init.trunc_normal_(mat, _mu, _std, _mu-2*_std, _mu+2*_std))

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