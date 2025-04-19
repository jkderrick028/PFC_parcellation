"""
Utilities for Dictionary Learning

author: Ana Luisa Pinho
email: agrilopi@uwo.ca

Compatibility: Python 3.9.20
"""

import torch
from joblib import Memory

import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import spectral_embedding
from sklearn.decomposition import dict_learning_online, sparse_encode
from sklearn.linear_model import MultiTaskLasso, MultiTaskElasticNet

from utils import normalize_rows


# ###################### FUNCTIONS  ####################################

def kmeans_dictionary(n_parcels, Y):
    """
    Create the initial dictionary using K-Means clustering

    Args:
        n_parcels (int): number of parcels
        Y (np.ndarray): fMRI data (n_obs, n_subjects * n_voxels)

    Returns:
        dict_init (np.ndarray): initial dictionary (n_parcels, n_obs)
    """

    kmeans = MiniBatchKMeans(n_clusters=n_parcels, random_state=0,
                             batch_size=200, n_init=10)
    kmeans = kmeans.fit(Y.T)

    dict_init_ = kmeans.cluster_centers_
    dict_init = (dict_init_.T / np.sqrt((dict_init_ ** 2).sum(1))).T
    similarity = np.exp(np.corrcoef(dict_init))
    embedding = spectral_embedding(similarity, n_components=1)
    order = np.argsort(embedding.T).ravel()
    dict_init = dict_init[order]

    return dict_init


def gaussian_dictionary(n_parcels, n_obs):
    """
    Initialize V matrix by random sampling from the normal distribution with
    additive noise

    Args:
        n_parcels (int): number of parcels
        n_obs (int): number of observations

    Returns:
        dict_init (np.ndarray): initial dictionary (n_parcels, n_obs)
    """

    # Generate V_init from the normal distribution --> (n_parcels, n_obs)
    dict_init = np.zeros((n_parcels, n_obs))
    dict_init = np.random.randn(n_parcels, n_obs)

    return dict_init


def online_loss(X, parcels, dictionary, alpha):
    """
    Compute the loss function for dictionary learning.

    Args:
        X (np.ndarray): fMRI data
                        (n_subjects * n_voxels, n_obs)
        parcels (np.ndarray): Individual parcellations
                              (n_subjects * n_voxels, n_parcels)
        dictionary (np.ndarray): dictionary of components
                                 (n_parcels, n_obs)

    Returns:
        loss (np.float): Value of the loss function.
    """

    loss = .5 * np.linalg.norm(X - parcels @ dictionary, 'fro') ** 2 + \
        alpha * np.sum(np.abs(parcels))

    return loss


def lasso_loss(X, X_pred, parcels, alpha):
    """
    Compute the loss function for Multitask Lasso.

    Args:
        X (np.ndarray): fMRI data
                        (n_obs, n_subjects * n_voxels)
        X_pred (np.ndarray): predicted X
                        (n_obs, n_subjects * n_voxels)
        parcels (np.ndarray): Individual parcellations
                              (n_parcels, n_subjects * n_voxels)
        alpha (np.float): regularization parameter

    Returns:
        loss (np.float): Value of the loss function.
    """

    # Compute the l1-l2 norm
    l1l2_norm = np.sum(np.sqrt(np.sum(parcels ** 2, axis=1)))

    # Compute loss
    loss = (1 / (2 * X.shape[0])) * np.linalg.norm(X - X_pred, 'fro') ** 2 + \
        alpha * l1l2_norm

    return loss


def enet_loss(X, X_pred, parcels, alpha, l1_ratio):
    """
    Compute the loss function for Multitask ElasticNet.

    Args:
        X (np.ndarray): fMRI data
                        (n_obs, n_subjects * n_voxels)
        X_pred (np.ndarray): predicted X
                        (n_obs, n_subjects * n_voxels)
        parcels (np.ndarray): Individual parcellations
                              (n_parcels, n_subjects * n_voxels)
        alpha (np.float): regularization parameter
        l1_ratio (np.float): ElasticNet mixing parameter

    Returns:
        loss (np.float): Value of the loss function.
    """

    # Compute the l1-l2 norm
    l1l2_norm = np.sum(np.sqrt(np.sum(parcels ** 2, axis=1)))

    # Compute loss
    loss = (1 / (2 * X.shape[0])) * np.linalg.norm(X - X_pred, 'fro') ** 2 + \
        alpha * l1_ratio * l1l2_norm + \
        (.5 * alpha * (1 - l1_ratio)) * np.linalg.norm(parcels, 'fro') ** 2

    return loss


def get_iparcel_dictlearning(Y, n_parcels=20, vinit_type = 'kmeans',
                             dict_init=None, method='online', alpha=.1,
                             l1_ratio=.5, write_dir='/tmp', n_iter=None):
    """
    Create the sparse-encoded individual parcellations

    Args:
        Y (np.ndarray): fMRI data per subject
                        (n_subjects, n_obs, n_voxels)
        dict_init (torch.Tensor or np.ndarray): Functional Profile
                                                (n_obs, n_parcels)
                                                default = None

    Returns:
        Ui (np.ndarray): Individual parcellations
                         (n_subjects, n_parcels, n_voxels)
        dictionary (np.ndarray): Dictionary of parcels
                                 (n_parcels, n_obs)
    """
    if n_iter is not None:
        print(n_iter)

    # Set nan's to 0 in the input data
    np.nan_to_num(Y, copy=False)

    # Retrieve number of subjects, observations and voxels
    n_subjects = Y.shape[0]
    n_obs = Y.shape[1]
    n_voxels = Y.shape[2]

    # Reshape input data --> (n_obs, n_subjects * n_voxels)
    y = Y.swapaxes(0, 1)
    y = np.reshape(y, (y.shape[0], -1))

    # Compute the initial dictionary across subjects
    # dict_init shape --> (n_parcels, n_obs)
    if vinit_type == 'kmeans':
        assert dict_init is None
        # Computes it only once and save it on the cache...
        mem = Memory(write_dir, verbose=0)
        dict_init = mem.cache(kmeans_dictionary)(n_parcels, y)
        # ... or compute it every time it passes here
        # dict_init = kmeans_dictionary(n_parcels, y)
    elif vinit_type == 'hbp':
        assert dict_init is not None
        if torch.is_tensor(dict_init):
            dict_init = dict_init.cpu().numpy()
        dict_init = dict_init.T
    else:
        # Generate Gaussian-type dictionaries
        assert vinit_type == 'gaussian'
        if dict_init is None:
            dict_init = gaussian_dictionary(n_parcels, n_obs)
        else:
            assert dict_init is not None
            pass

    if method == 'online':
        parcels, dictionary = dict_learning_online(
            y.T,
            n_components=n_parcels,
            alpha=alpha,
            return_code=True,
            dict_init=dict_init,
            batch_size=256,
            method='cd',
            shuffle=True,
            positive_code=True,
        )
    elif method in ['mt_lasso', 'mt_enet']:
        # parcels shape --> (n_subjects * n_voxels, n_parcels)
        parcels = np.zeros((y.shape[1], n_parcels))
        # y_pred shape --> (n_obs, n_subjects * n_voxels)
        y_pred = np.zeros(y.shape)
        if method == 'mt_lasso':
            clf = MultiTaskLasso(alpha=alpha)
        else:
            assert method == 'mt_enet'
            clf = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        for i in np.arange(n_voxels):
            # x shape --> (n_obs, n_subjects)
            x = y[:, i: i + n_subjects * n_voxels: n_voxels]
            parcels[i: i + n_subjects * n_voxels: n_voxels] = \
                clf.fit(dict_init.T, x).coef_
            y_pred[:, i: i + n_subjects * n_voxels: n_voxels] = \
                clf.predict(dict_init.T)
    else:
        assert method == 'sparse'
        parcels = sparse_encode(
            y.T, dict_init, alpha=alpha, max_iter=100, n_jobs=1,
            check_input=True, positive=True)
        dictionary = dict_init

    # Compute Loss
    if method in ['online', 'sparse']:
        err = online_loss(y.T, parcels, dictionary, alpha)
    elif method == 'mt_lasso':
        err = lasso_loss(y, y_pred, parcels.T, alpha)
    else:
        assert method == 'mt_enet'
        err = enet_loss(y, y_pred, parcels.T, alpha, l1_ratio)

    # Sparsity
    flat_U = np.ravel(parcels.T)
    sparsity = np.mean(flat_U == 0)
    print('Sparsity of U across subjects for ' + method + ' is:', sparsity)

    # Reshape parcels --> (n_subjects, n_parcels, n_voxels)
    Ui = parcels.T.reshape(n_parcels, parcels.shape[0] // n_voxels, n_voxels)
    Ui = np.swapaxes(Ui, 0, 1)

    # Normalize Ui's 
    # (for a given subject and each voxel, the sum of the parcels is 1)
    for i in np.arange(n_subjects):
        Ui[i, :, :] = normalize_rows(Ui[i, :, :].T).T

    # Reshape initial dictionary -- > (n_obs, n_parcels)
    dict_init = np.swapaxes(dict_init, 0, 1)

    return Ui, dict_init, err
