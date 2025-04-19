"""
Dual regression to estimate individual parcellations using
the Functional-Fusion framework

author: Ana Luisa Pinho
email: agrilopi@uwo.ca

created: January 13, 2024
last update: December 2024

Compatibility: Python 3.9.20
"""

import torch as pt
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

import matplotlib.pyplot as plt

import numpy as np
from scipy.special import softmax
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FastICA

import Functional_Fusion.dataset as ds
import HierarchBayesParcel.full_model as fm

from global_config import MODEL_DIR, BASE_DIR, DEVICE
from utils_dictionary import kmeans_dictionary
from utils import pretrained_arrmodel, normalize_rows, plot_multi_flat

# ###################### FUNCTIONS  ####################################


def get_hbp_v(ar_model, atlas, data, info):
    """
    Get V for a given emission model of HBP

    Returns:
    V_em (torch.Tensor): (n_obs, n_parcels)
    """
    cond_v = info['cond_num_uni'].to_numpy()
    part_v = np.array([int(i[-1]) for i in info['sess']])
    # part_v = tinfo['half'].to_numpy()
    sub_ind = np.arange(data.shape[0])
    _, _, M = fm.get_indiv_parcellation(ar_model, atlas, [data], [cond_v],
                                        [part_v], [sub_ind])
    V_em = M.emissions[0].V

    return V_em


def kmeans_vinit(n_parcels, X):
    """
    Create the initial dictionary using K-Means clustering

    Args:
        n_parcels (int): number of parcels
        X (np.ndarray): fMRI data (n_subjects, n_obs, n_voxels)

    Returns:
        dict_init (np.ndarray): initial dictionary (n_parcels, n_obs)
    """

    # Set nan's to 0 in the input data
    np.nan_to_num(X, copy=False)

    # Reshape input data --> (n_obs, n_subjects * n_voxels)
    Y = X.swapaxes(0, 1)
    Y = np.reshape(Y, (Y.shape[0], -1))

    # Compute dictionary with kmeans
    dict_init = kmeans_dictionary(n_parcels, Y)

    return dict_init.T


def group_ica(Y, n_components=None, prior_type='prob'):
    """
    Dual regression to estimate individual parcellations

    Args:
        Y (np.ndarray): Individual fMRI data (n_subjects, n_obs, n_voxels)
        n_components (np.int): Number of components to use.
                               If None is passed, all are used.
        prior_type (str): Type of prior to use in the model. 
                          Accepts only 'prob' or 'logpi'. 
                          (default: 'prob')

    Returns:
        Ug (torch.Tensor or np.ndarray): Group map (n_voxels, n_parcels)
    """

    # Set nan's to 0 in the input data
    np.nan_to_num(Y, copy=False)

    # Reshape input data --> (n_voxels, n_subjects * n_obs)
    Y = np.reshape(Y, (Y.shape[0] * Y.shape[1], -1))
    Y = Y.swapaxes(0, 1)

    # Compute ICA
    transformer = FastICA(n_components=n_components, random_state=0,
                          whiten='unit-variance')
    Y_transformed = transformer.fit_transform(Y)

    # Normalization (compute probability distribution of group map)
    if prior_type == 'logpi':
        prob_Ug = softmax(Y_transformed)
    else:
        # If already a probability, make sure the sum along each row is 1
        assert prior_type == 'prob'

        prob_Ug = normalize_rows(Y_transformed)

    return prob_Ug


def get_iparcel_dualreg(Y, Ug=None, V=None, prior_type='prob'):
    """
    Dual Regression to estimate individual parcellations

    Args:
        Y (np.ndarray): Individual fMRI data 
                        (n_subjects, n_obs, n_voxels)
        Ug (torch.Tensor or np.ndarray): Group map 
                                         (n_voxels, n_parcels)
        V (torch.Tensor or np.ndarray): Functional Profile 
                                        (n_obs, n_parcels)
        prior_type (str): Type of prior to use in the model. 
                          Accepts only 'prob' or 'logpi'. 
                          (default: 'prob')

    Returns:
        Ui (np.ndarray): Individual parcellation
        (n_subjects, n_parcels, n_voxels)
    """

    # Set nan's to 0 in the input data
    np.nan_to_num(Y, copy=False)

    if Ug is not None:
        # Move the tensor to CPU and convert it into a NumPy array
        if pt.is_tensor(Ug):
            Ug = Ug.cpu().numpy()

        # Set nan's to 0 in the atlas
        np.nan_to_num(Ug, copy=False)

        # Normalization (compute probability distribution of group map)
        if prior_type == 'logpi':
            prob_Ug = softmax(Ug)
        else:
            # Otherwise, make sure the sum along each row is 1
            assert prior_type == 'prob'

            prob_Ug = normalize_rows(Ug)

    if V is not None and pt.is_tensor(V):
        V = V.cpu().numpy()

    # For every individual:
    Ui = []
    for y in Y:       
        if V is None:
            assert Ug is not None
            # Initialize V
            V = np.empty((y.shape[0], prob_Ug.shape[1]))
            # Fit the OLS to generate V --> (n_obs, n_parcels)
            reg_ols = LinearRegression()
            V = reg_ols.fit(prob_Ug, y.T).coef_

        # Normalize V along the columns (sum to 1 over the n_obs)
        V_normalized = V / np.linalg.norm(V, axis=0, keepdims=True)
        # Fit the Non-Negative Least Squares to compute Ui -- >
        # (n_parcels, n_voxels)
        ui = np.hstack(Parallel(n_jobs=-1)(
            delayed(lambda y_p: nnls(V_normalized, y_p, 
                                     maxiter=300 * V_normalized.shape[1], 
                                     atol=1e-06)[0][:, np.newaxis])
            (y[:, p]) for p in tqdm(range(y.shape[1]), desc='Computing NNLS')
        ))

        # Normalize ui, the individual parcellations for each subject
        ui_normalized = normalize_rows(ui.T)
        ui_normalized = ui_normalized.T

        # Append
        Ui.append(ui_normalized)

    return np.array(Ui)


# ######################### INPUTS ######################################

# # Load the atlas
atlas_name = 'MNISymC3'
sym_type = 'asym'
model_name = f'/Models_03/asym_Md_space-MNISymC3_K-17'

train_sess = 'ses-s1'

# Define some paths (cross-platform valid)
# home_dir = str(Path.home())
main_dir = Path.cwd()
output_dir = Path(str(Path(main_dir, 'dual_regression')))

# ########################## RUN #########################################

if __name__ == "__main__":

    # Load Arrangement Model
    atlas, _, U, _, ar_model = pretrained_arrmodel(
        atlas_name, MODEL_DIR, model_name, DEVICE, sym_type)

    # Load the individual localizing data from the Functional-Fusion framework
    tdata, tinfo, _ = ds.get_dataset(
        BASE_DIR, 'MDTB', atlas=atlas.name, subj=None, sess=train_sess,
        type='CondAll')

    # Set number of parcels
    n_parcels = int(model_name[-2:])

    # Create output folder, if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute GroupICA of input data
    U_ica = group_ica(tdata, n_components=n_parcels)

    # Estimate individual parcellations using HBP group atlas
    Ui_uhbp = get_iparcel_dualreg(tdata, Ug=U.T)

    # Estimate individual parcellations using Group ICA
    Ui_uica = get_iparcel_dualreg(tdata, Ug=U_ica)

    # Estimate individual parcellations using HBP V initialization
    hbp_vinit = get_hbp_v(ar_model, atlas, tdata, tinfo)
    Ui_vhbp = get_iparcel_dualreg(tdata, V=hbp_vinit)

    # Estimate individual parcellations using Kmeans-V initialization
    km_vinit = kmeans_vinit(n_parcels, tdata)
    Ui_vkm = get_iparcel_dualreg(tdata, V=km_vinit)

    # Save individual parcellations
    np.save(str(Path(output_dir, 'iparcel_mdtb_' + train_sess +
                     '_mni_asym-k17_dual-regression.npy')),
            Ui_vkm)

    # Visualization: plot Ui, i.e. the probabilistic individual 
    # parcellations
    plt.figure(figsize=(20, 20))
    plot_multi_flat(
        Ui_vkm, atlas_name, grid=(6, 4), cmap='tab20', dtype='prob',
        titles=["subj_{}".format(i+1) for i in range(Ui_vkm.shape[0])],
        fig_path=str(Path(output_dir, 'iparcel_mdtb_' + train_sess +
                          '_mni_sym-k20_dual-regression.png')))
    plt.show()
