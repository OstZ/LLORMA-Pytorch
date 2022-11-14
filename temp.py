import torch
import numpy as np
from torch.nn import functional as F
from configs import *
# from model import pretrian_model
from batch import BatchManager

from pre_trainer import get_feats

from sklearn.preprocessing import normalize

class handcraft_model():
    def __init__(self,lr , batch_manager):
        self.n_row = batch_manager.n_user
        self.n_col = batch_manager.n_item
        self.mu = batch_manager.mu
        self.std = batch_manager.std
        self.lr = lr
        self.u = np.array(self._create_p_or_q_variable(self.n_row, PRE_RANK, self.mu, self.std).detach())
        self.v = np.array(self._create_p_or_q_variable(self.n_col, PRE_RANK, self.mu, self.std).detach())
    def _create_p_or_q_variable(self, n, rank, mu, std):
        # 按特定分布初始化p,q
        _mu = np.sqrt(mu / rank)
        _std = np.sqrt((np.sqrt(mu * mu + std * std) - mu) / rank)
        mat = torch.empty(n, rank)
        return torch.as_tensor(torch.nn.init.trunc_normal_(mat, _mu, _std,
                                                           _mu - 2 * _std,
                                                           _mu + 2 * _std))
    def train(self,u,i,r):
        tmp_u = self.u
        tmp_v = self.v
        n = len(u)
        for idx in range(len(u)):
            u_idx = int(u[idx])
            i_idx = int(i[idx])
            self.u[u_idx] -= 1/n*self.lr*((np.dot(tmp_v[i_idx],tmp_u[u_idx].T) - 2*r[idx])*tmp_v[i_idx] + 0.001*self.u[u_idx])
        tmp_u = self.u
        tmp_v = self.v
        for idx in range(len(i)):
            u_idx = int(u[idx])
            i_idx = int(i[idx])
            self.v[i_idx] -= 1/n*self.lr*((np.dot(tmp_u[u_idx],tmp_v[i_idx].T) - 2*r[idx])*tmp_u[u_idx] + 0.001*self.v[i_idx])
    def eval(self,u,i,r):
        u_lookup = F.embedding(torch.from_numpy(u).long(), torch.tensor(self.u))
        i_lookup = F.embedding(torch.from_numpy(i).long(), torch.tensor(self.v))
        r_hat = torch.sum(torch.multiply(u_lookup, i_lookup), 1)

        r = torch.tensor(r,dtype = torch.float)
        MSE = torch.nn.MSELoss()
        rmse = torch.sqrt(MSE(r_hat, r))
        return rmse





class pretrian_model(torch.nn.Module):
    def __init__(self,batch_manager):
        super(pretrian_model,self).__init__()
        self.n_row = batch_manager.n_user
        self.n_col = batch_manager.n_item
        self.mu = batch_manager.mu
        self.std = batch_manager.std
        self.u_feat = self._create_p_or_q_variable(self.n_row, PRE_RANK, self.mu, self.std).requires_grad_(True)
        self.i_feat = self._create_p_or_q_variable(self.n_col, PRE_RANK, self.mu, self.std).requires_grad_(True)

    def _create_p_or_q_variable(self,n, rank, mu, std):
        # 按特定分布初始化p,q
        _mu = np.sqrt(mu / rank)
        _std = np.sqrt((np.sqrt(mu * mu + std * std) - mu) / rank)
        mat = torch.empty(n,rank)
        return torch.as_tensor(torch.nn.init.trunc_normal_(mat, _mu, _std, _mu-2*_std, _mu+2*_std))

    def forward(self, u, i, r):
        u_lookup = F.embedding(u,self.u_feat)
        i_lookup = F.embedding(i,self.i_feat)
        r_hat = torch.sum(torch.multiply(u_lookup,i_lookup),1)

        MSE = torch.nn.MSELoss()
        rmse = torch.sqrt(MSE(r_hat, r))

        reg_loss = torch.add(torch.sum(torch.square(self.u_feat))
                            , torch.sum(torch.square(self.i_feat)))
        loss = MSE(r_hat, r) + PRE_LAMBDA*reg_loss
        return loss,rmse
def pre_test(model,batchmanager):
    test_data = torch.from_numpy(batchmanager.test_data)
    u = torch.as_tensor(test_data[:, 0], dtype=torch.long)
    i = torch.as_tensor(test_data[:, 1], dtype=torch.long)
    r = torch.as_tensor(test_data[:, 2], dtype=torch.float32)
    model.eval()
    test_loss, rmse = model(u, i, r)
    print("test loss:{} RMSE:{}".format(test_loss, rmse))
    return rmse



if __name__ == "__main__":
    # rmse = []
    batchmanager = BatchManager('movielens-100k')
    # model = pretrian_model(batchmanager)
    # optomizer = torch.optim.Adam([model.u_feat,model.i_feat],lr=PRE_LEARNING_RATE, weight_decay=1e-5)
    # train_data = torch.from_numpy(batchmanager.train_data)
    # u = torch.as_tensor(train_data[:,0],dtype=torch.long)
    # i = torch.as_tensor(train_data[:,1],dtype=torch.long)
    # r = torch.as_tensor(train_data[:,2],dtype=torch.float32)
    # for epoch in range(3000):
    #     loss_train,_ = model(u, i, r)
    #     optomizer.zero_grad()
    #     loss_train.backward()
    #     optomizer.step()
    #     RMSE = pre_test(model,batchmanager)
    #     rmse.append(np.array(RMSE.detach()))
    # np.save("rmse",rmse)
    # train_data = batchmanager.train_data
    # u = train_data[:, 0]
    # i = train_data[:, 1]
    # r = train_data[:, 2]
    # test_data = batchmanager.test_data
    # test_u = test_data[:, 0]
    # test_i = test_data[:, 1]
    # test_r = test_data[:, 2]
    # model_hand = handcraft_model(0.01,batchmanager)
    # for idx in range(500):
    #     model_hand.train(u,i,r)
    #     rmse = model_hand.eval(test_u, test_i, test_r)
    #     print("epoch:{} rmse:{}".format(idx, rmse))
    # train_data = np.array(batchmanager.train_data)
    # train_user_ids = train_data[:, 0].astype(np.int64)
    # train_item_ids = train_data[:, 1].astype(np.int64)
    f_u,f_i = get_feats("movielens-1m")
    # user_id = 0
    # item_id = 50
    # k = np.multiply(f_u[user_id][train_user_ids],
    #                 f_i[item_id][train_item_ids])
    # n_f_u = normalize(f_u)
    # cos = np.matmul(n_f_u,n_f_u.T)
    # print(cos[10:20][:10])



