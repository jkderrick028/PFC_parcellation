"""
Utilities for Individual-Parcellation Pipelines

authors: Da Zhi, Ana Luisa Pinho

Compatibility: Python 3.9.20
"""

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.evaluation as ev
import HierarchBayesParcel.util as ut
import Functional_Fusion.atlas_map as am
from FusionModel.util import plot_data_flat


def pretrained_arrmodel(atlas_name, model_dir, model_name, device, sym_type,
                        prior_type='logpi'):
    """
    Define arrangement model
    """
    atlas, ainf = am.get_atlas(atlas_name)

    atlas_fname = model_dir + model_name
    U, minfo = ut.load_group_parcellation(atlas_fname, device=device)
    ar_model = ar.build_arrangement_model(
        U, prior_type=prior_type, atlas=atlas, sym_type=sym_type)

    return atlas, ainf, U, minfo, ar_model


def prob_arrmodel(atlas_name, atlas_fname, sym_type, prior_type='prob'):
    """
    Define arrangement model
    """
    atlas, ainf = am.get_atlas(atlas_name)
    U = atlas.read_data(atlas_fname)
    U = U.T
    ar_model = ar.build_arrangement_model(
        U, prior_type=prior_type, atlas=atlas, sym_type=sym_type)

    return atlas, ainf, U, ar_model


def normalize_rows(X, constant=1e-10):
    """
    Normalize rows of a matrix

    Args:
        X : ndarray, shape (p, k)
            The input matrix, where p is the number of rows 
            (voxels) and k is the number of columns (parcels).
        constant : float 
            A small constant to avoid division by zero

    Returns:
        X_normalized : ndarray, shape (p, k)
            The normalized matrix with each p row summing to 1.
    """

    # Calculate the sum of each row
    row_sums = np.sum(X, axis=1)

    # Find rows where the sum is 0
    zero_rows = row_sums == 0

    if np.any(zero_rows):
        # Add the constant to all elements of rows where the sum is 0
        X[zero_rows, :] += constant

        # Recalculate the sum of each row
        row_sums = np.sum(X, axis=1)

    # Reshape row_sums to enable broadcasting
    row_sums = row_sums[:, np.newaxis]

    # Divide each row by its sum
    X_normalized = X / row_sums

    return X_normalized


def cv_cosine_ols(test_data, U_hats, adjusted=True, coserr_type='average'):
    """
    Cross-validation scheme for cosine-error evaluation using 
    dual-regression and dictionary-learning parcellations, where V 
    for n-1 subjects of the test set is estimated with OLS regression

    Args:
        test_data : ndarray, shape (n_subjects, n_obs, n_voxels)
            fMRI data per subject
        U_hats : ndarray, shape (n_subjects, n_parcels, n_voxels)
            Individual parcellations

    Returns:
        cosine_err : ndarray, shape (n_subjects)
            Cosine error for each subject
    """

    # Set nan's to 0 in the input data
    np.nan_to_num(test_data, copy=False)

    n_subjects = test_data.shape[0]
    cosine_err = np.empty((n_subjects))
    subj = np.arange(n_subjects)
    for s in subj:
        # ############## ESTIMATE V FOR THE TEST TASKS #################
        # Remove left-out subject for validation from the test tasks
        Z = test_data[subj != s, :, :]
        Uz = U_hats[subj != s, :, :]

        # Reshape data and parcellations --> (n_obs, n_subjects * n_voxels)
        Z_reshaped = Z.swapaxes(0, 1)
        Uz_reshaped = Uz.swapaxes(0, 1)

        Z_reshaped = np.reshape(Z_reshaped, (Z_reshaped.shape[0], -1))
        Uz_reshaped = np.reshape(Uz_reshaped, (Uz_reshaped.shape[0], -1))
 
        # Initialize Vz
        Vz = np.empty((Z_reshaped.shape[0], Uz_reshaped.shape[0]))
        # Fit the OLS to generate V --> (n_obs, n_parcels)
        Vz = np.linalg.pinv(Uz_reshaped.T).dot(Z_reshaped.T).T

        # ################ COMPUTE COSINE ERROR ########################
        dat = torch.from_numpy(test_data[subj == s, :, :]).to(torch.float32)
        Vz = torch.from_numpy(Vz).to(torch.float32)
        U_test = torch.from_numpy(U_hats[subj == s, :, :]).to(torch.float32)

        a = ev.cosine_error(dat, Vz, U_test, adjusted=adjusted, 
                            type=coserr_type)
        cosine_err[s] = a

    return cosine_err


def plot_multi_flat(data, atlas, grid, cmap='tab20b', dtype='label',
                    cscale=None, titles=None, colorbar=False, fig_path=None):
    """ Plot multiple flatmaps in a grid

    Args:
        data: the input parcellations, shape(N, K, P) where N indicates
              the number of parcellations, K indicates the number of
              parcels, and P is the number of vertices.
        atlas: the atlas name used to plot the flatmap
        grid: the grid shape of the subplots
        cmap: the colormap used to plot the flatmap
        dtype: the data type of the input data, 'label' or 'prob'
        cscale: the color scale used to plot the flatmap
        titles: the titles of the subplots
        colorbar: whether to plot the colorbar
        fig_path: path of output figure, default format is png

    Returns:
        The plt figure plot
    """

    if isinstance(data, np.ndarray):
        n_subplots = data.shape[0]
    elif isinstance(data, list):
        n_subplots = len(data)

    if not isinstance(cmap, list):
        cmap = [cmap] * n_subplots

    for i in np.arange(n_subplots):
        plt.subplot(grid[0], grid[1], i + 1)
        plot_data_flat(data[i], atlas,
                       cmap=cmap[i],
                       dtype=dtype,
                       cscale=None,
                       render='matplotlib',
                       colorbar=(i == 0) & colorbar)

        plt.title(titles[i])
        plt.tight_layout()

    if isinstance(fig_path, str):
        plt.savefig(fig_path)


def convert_hard_to_prob(U, strength=7.0):
    ''' Convert a hard parcellation to probabilistic

    Args:
        U (np.ndarray): P-long vector of hard parcellation
        strength (float): the strength of the parcellation (prior)

    Returns:
        the probabilistic parcellation, either a (K, P) group
        map or a (n_subj, K, P) individual map. In whichever case,
        the sum along dimension K equals to 1, making it a
        probabilistic parcellation.

    Notes:
        In some exisiting hard parcellations, a voxel may have a
        label 0 to indicate unassigned parcel. For these voxels,
        we give a flat distribution - so that the probability of
        such a voxel to assign one parcel is 1/K.
    '''
    assert np.all((U >= 0) & (U.astype(int) == U)), \
        "The input U must be non-negative integer numpy array!"
    _, U = np.unique(U, return_inverse=True)
    K = np.unique(U).size

    if U.ndim == 1:
        logpi = ar.expand_mn_1d(U, K) * strength
        # Set parcel 0 to unassigned
        logpi = logpi[1:, :] if np.any(np.unique(U) == 0) else logpi
        return torch.softmax(logpi, dim=0)
    elif U.ndim == 2:
        logpi = ar.expand_mn(U, K) * strength
        # Set parcel 0 to unassigned
        logpi = logpi[:, 1:, :] if np.any(np.unique(U) == 0) else logpi
        return torch.softmax(logpi, dim=1)
    else:
        raise ValueError('The input U must be (P,) or (n_subj, P) integer ndarray!')


def load_batch_fit(fname, device=None):
    """ Loads a batch of fitted models
    Args:
        fname (str): File directory
        device: model model to which device, 'cpu', 'cuda'..
    Returns:
        info: Data Frame with information
        models: List of models
    """
    info = pd.read_csv(fname + '.tsv', sep='\t')
    with open(fname + '.pickle', 'rb') as file:
        models = pickle.load(file)

    if device is not None:
        for m in models:
            m.move_to(device)

    return info, models


def load_batch_best(fname, device=None):
    """ Loads a batch of model fits and selects the best one

    Args:
        fname (str): File name
        device: model model to which device, 'cpu', 'cuda'..

    Returns:
        info_reduced: Data Frame with reduced information
        best_model: the model with highest likelihood
    """
    info, models = load_batch_fit(fname)

    j = info.loglik.argmax()

    best_model = models[j]
    if device is not None:
        best_model.move_to(device)

    info_reduced = info.iloc[j]
    return info_reduced, best_model