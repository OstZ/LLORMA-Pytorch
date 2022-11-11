import random
import torch
import numpy as np
from .batch import BatchManager
from .configs import *
from .model import pretrian_model

def _validate(session, batch_manager, models):
    valid_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['u']: batch_manager.valid_data[:, 0],
            models['i']: batch_manager.valid_data[:, 1],
            models['r']: batch_manager.valid_data[:, 2]
        })

    test_rmse = session.run(
        models['rmse'],
        feed_dict={
            models['u']: batch_manager.test_data[:, 0],
            models['i']: batch_manager.test_data[:, 1],
            models['r']: batch_manager.test_data[:, 2]
        })

    return valid_rmse, test_rmse

def get_feats(kind,use_cache=True):
    if use_cache:
        try:
            feat_u = np.load('data/{}-feat_u.npy'.format(kind))
            feat_v = np.load('data/{}-feat_v.npy'.format(kind))
            return feat_u, feat_v
        except:
            print('>> There is no cached feat_u and feat_v.')
    batch_manager = BatchManager(kind)
    models = pretrian_model(batch_manager)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
