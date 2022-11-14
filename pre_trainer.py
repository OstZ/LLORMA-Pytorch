import random
import torch
import numpy as np
from batch import BatchManager
from configs import *
from model import pretrian_model

def _eval(batch_manager, model):
    '''eval the pretrain model'''
    model.eval()
    valid_data = batch_manager.valid_data
    _,valid_rmse = model(valid_data[:,0],
                       valid_data[:,1],
                       torch.as_tensor(valid_data[:, 2],
                                       dtype=torch.float32))

    test_data = batch_manager.test_data
    _,test_rmse = model(test_data[:,0],
                      test_data[:,1],
                      torch.as_tensor(test_data[:, 2],
                                      dtype=torch.float32))

    return valid_rmse, test_rmse

def get_feats(kind,use_cache=True):
    '''
       get pretrained latent feature to meature the distance
       between users and among items
    '''
    if use_cache:
        try:
            feat_u = np.load('features/{}-feat_u.npy'.format(kind))
            feat_i = np.load('features/{}-feat_i.npy'.format(kind))
            return feat_u, feat_i
        except:
            print('>> There is no cached feat_u and feat_i.')
    batch_manager = BatchManager(kind)
    model = pretrian_model(batch_manager)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #variable to record rmse and features
    min_valid_rmse = float(2.0)
    min_valid_iter = 0
    final_test_rmse = float(2.0)
    threshold = PRE_THRESHOLD


    #load data
    train_data = batch_manager.train_data
    u = train_data[:, 0]
    i = train_data[:, 1]
    r = torch.as_tensor(train_data[:, 2],dtype=torch.float32)

    #initialize optimizer
    opt1 = torch.optim.SGD([model.get_u_feat()], lr=PRE_LEARNING_RATE, momentum=0.9)
    opt2 = torch.optim.SGD([model.get_i_feat()], lr=PRE_LEARNING_RATE, momentum=0.9)

    #train
    model.train()
    for idx in range(10000):
        loss, _ = model(u, i, r)
        opt1.zero_grad()
        loss.backward()
        opt1.step()

        loss, rmse = model(u, i, r)
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        valid_rmse, test_rmse = _eval(batch_manager,model)
        if idx > min_valid_iter + 100:
            break
        if min_valid_rmse > valid_rmse + PRE_THRESHOLD:
            min_valid_rmse = valid_rmse
            min_valid_iter = idx
            final_test_rmse = test_rmse
            #save corresponding features
            # np.save('features/{}-feat_u.npy'.format(kind),
            #         np.array(model.get_u_feat().detach()))
            # np.save('features/{}-feat_i.npy'.format(kind),
            #         np.array(model.get_i_feat().detach()))

        print('>> ITER:',
              "{:3d}".format(idx), "train: {:3f} val: {:3f} test: {:3f} / {:3f}".format(
                rmse, valid_rmse, test_rmse, final_test_rmse))

    #load features
    feat_u = np.load('features/{}-feat_u.npy'.format(kind))
    feat_i = np.load('features/{}-feat_i.npy'.format(kind))

    return feat_u,feat_i
