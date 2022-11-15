import random

import numpy as np
import torch
from sklearn.preprocessing import normalize

from configs import *


def _init_anchor_points(train_data, row_k, col_k):
    train_user_ids = train_data[:, 0].astype(np.int64)
    train_item_ids = train_data[:, 1].astype(np.int64)

    anchor_idxs = []
    while len(anchor_idxs) < N_ANCHOR:
        anchor_idx = random.randint(0, train_data.shape[0] - 1)
        if anchor_idx in anchor_idxs:
            continue

        anchor_row = train_data[anchor_idx]
        user_id = int(anchor_row[0])
        item_id = int(anchor_row[1])

        k = np.multiply(row_k[user_id][train_user_ids],
                        col_k[item_id][train_item_ids])
        #drop anchor that too far away from train pairs
        sum_a_of_anchor = np.sum(k)
        if sum_a_of_anchor < 1:
            continue

        print('>> %10d\t%d' % (anchor_idx, sum_a_of_anchor))
        anchor_idxs.append(anchor_idx)

    return anchor_idxs


def _get_distance_matrix(latent):
    '''
        get arccos matrix, store distance between vectors
        shape of (len(latent),len(latent))
    '''
    _normalized_latent = normalize(latent, axis=1)
    # print(_normalized_latent.shape)

    cos = np.matmul(_normalized_latent, _normalized_latent.T)
    cos = np.clip(cos, -1, 1)
    d = np.arccos(cos)
    assert np.count_nonzero(np.isnan(d)) == 0
    return d


def _get_k_from_distance(d):
    '''
        get kernel matrix,store K_h(i,j) in corresponding position
        shape of d
    '''
    #define the mask
    m = np.zeros(d.shape)
    m[d < 0.8] = 1
    #Epanechnikov kernel with indicator
    return np.multiply(np.subtract(np.ones(d.shape), np.square(d)), m)


def _get_ks_from_latents(row_latent, col_latent):
    '''
    :param row_latent: latent features of u
    :param col_latent: latent features of i
    :return: two kernel matrix
    '''

    # for i in range(row_latent.shape[0]):
    #     print(row_latent[i][:4])
    #
    # assert False
    row_d = _get_distance_matrix(row_latent)
    col_d = _get_distance_matrix(col_latent)

    row_k = _get_k_from_distance(row_d)
    col_k = _get_k_from_distance(col_d)

    return row_k, col_k


class AnchorManager:
    def __init__(
            self,
            batch_manager,
            feat_u,
            feat_i, ):

        train_data = np.array(batch_manager.train_data.detach())

        latent_u = feat_u
        latent_i = feat_i

        u_k, i_k = _get_ks_from_latents(latent_u, latent_i)

        anchor_idxs = _init_anchor_points(train_data, u_k, i_k)
        assert len(anchor_idxs) == N_ANCHOR
        # print(anchor_idxs)
        anchor_points = train_data[anchor_idxs]

        self.train_data = train_data
        self.test_data = batch_manager.test_data

        self.anchor_idxs = anchor_idxs
        self.anchor_points = anchor_points

        self.u_k = u_k
        self.i_k = i_k

    def _get_k(self, anchor_idx, data):
        row_k = self.u_k
        col_k = self.i_k
        anchor_point = self.anchor_points[anchor_idx]

        user_id = int(anchor_point[0])
        item_id = int(anchor_point[1])

        user_ids = data[:, 0]
        item_ids = data[:, 1]

        return np.multiply(row_k[user_id][user_ids], col_k[item_id][item_ids])

    def get_train_k(self, anchor_idx):
        return self._get_k(anchor_idx, self.train_data)

    def get_test_k(self, anchor_idx):
        return self._get_k(anchor_idx, self.test_data)
