"""
Data-driven models with different regularization schemes to estimate
individual parcellations using the Functional-Fusion framework

author: Ana Luisa Pinho
email: agrilopi@uwo.ca

created: January 29, 2024
last update: July 2024

Compatibility: Python 3.11.5
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import Functional_Fusion.dataset as ds
import HierarchBayesParcel.HierarchBayesParcel.full_model as fm

from global_config import MODEL_DIR, BASE_DIR, DEVICE
from utils_dictionary import get_iparcel_dictlearning
from utils import pretrained_arrmodel, plot_multi_flat


# ###################### FUNCTIONS  ####################################

def get_hbp_v(ar_model, atlas, data, info):
    """
    Get V for a given emission model of HBP

    Returns:
    V_em (torch.Tensor): (n_obs, n_parcels)
    """
    cond_v = info['cond_num_uni'].to_numpy()
    part_v = np.array([int(i[-1]) for i in info['sess']])
    sub_ind = np.arange(data.shape[0])
    _, _, M = fm.get_indiv_parcellation(ar_model, atlas, [tdata], [cond_v],
                                        [part_v], [sub_ind])
    V_em = M.emissions[0].V

    return V_em


def adjust_spines(ax, spines, n_offset):
    for loc, spine in ax.spines.items():
        if loc in spines:
            # outward by n_offset points
            spine.set_position(('outward', n_offset))

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')


def plot_histogram(sorted_gerrs, kerr, hbperr, min_xlabel, max_xlabel, n_bins,
                   write_dir, n_iter, method, desc):
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(left=.1, right=.98, bottom=.2, wspace=.075)
    bin_edges = np.linspace(min_xlabel, max_xlabel, n_bins).astype('int')
    # Create a histogram with a density estimate line
    sns.histplot(sorted_gerrs, kde=True, bins=bin_edges)
    ax.plot(kerr, 1., marker='o', markersize=8, color='tab:olive',
            label='K-Means error')
    ax.plot(hbperr, 1., marker='o', markersize=8, color='tab:pink',
            label='HBP error')
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    # Set the limits of the y-axis to ensure the dot is fully visible
    plt.ylim(-1, None)
    # Manually set the tick locations and labels on the x-axis
    plt.xticks(bin_edges, bin_edges)
    # Rotate x-tick labels
    ax.tick_params(axis='x', rotation=25)
    # Add offset in x-axis
    adjust_spines(ax, ['left'], 0)
    adjust_spines(ax, ['bottom'], 10)
    # Add a label to the x-axis
    plt.xlabel('Error')
    # Add legend
    plt.title('Histogram + Density Plot w/ iter = ' + str(n_iter) + \
              ' for method ' + method)
    plt.legend(frameon=False)
    # Save figuremt_enet
    plt.savefig(str(Path(write_dir, 'histogram-plot' + desc + '.png')))


# ######################### INPUTS ######################################

## Load the atlas
atlas_name = 'MNISymC3'
sym_type = 'asym'
model_name = f'/Models_03/asym_Md_space-MNISymC3_K-17'

# Method's parameters
methods = {'online': 'hbp', 'mt_lasso': 'hbp', 'mt_enet': 'hbp', 
           'sparse': 'hbp'} # online, sparse, mt_lasso, mt_enet
alpha = .1
l1_ratio = .5 # only for multitask_elasticnet
train_sess='ses-s1' # for mdtb: 'all', 'ses-s1' or 'ses-s2'

# Define some paths (cross-platform valid)
# home_dir = str(Path.home())
main_dir = Path.cwd()

# ######################### RUN #########################################

if __name__ == "__main__":
    
    # Load Arrangement Model
    atlas, _, _, _, ar_model = pretrained_arrmodel(
        atlas_name, MODEL_DIR, model_name, DEVICE, sym_type)
    
    # Load the individual localizing data from the Functional-Fusion
    # framework
    tdata, tinfo, _ = ds.get_dataset(
        BASE_DIR, 'MDTB', atlas=atlas.name, subj=None, sess=train_sess,
        type='CondAll')

    # Set number of clusters
    n_clusters = int(model_name[-2:])

    # # ############# ESTIMATE AND PLOT INDIVIDUAL PARCELLATIONS ###########

    for method, vtype in methods.items():

        method_dir = Path(str(Path(main_dir, method)))
        # Create output folder, if it does not exist
        method_dir.mkdir(parents=True, exist_ok=True)

        if vtype == 'kmeans':
            # Compute parcellations using K-means V_init
            U_i, _, _ = get_iparcel_dictlearning(
                tdata, vinit_type=vtype, n_parcels=n_clusters, method=method,
                alpha=alpha)
        elif vtype == 'hbp':
            # Compute parcellations using HBP V_init
            hbp_vinit = get_hbp_v(ar_model, atlas, tdata, tinfo)
            U_i, _, _ = get_iparcel_dictlearning(
                tdata, vinit_type=vtype, dict_init=hbp_vinit, n_parcels=n_clusters,
                method=method, alpha=alpha)
        else:
            assert vtype ==  'gaussian'
            # Compute parcellations using min Gaussian V_init
            gauss_err = np.load(Path(str(Path(
                method_dir, 'errs_mdtb_' + train_sess + '_' + method + \
                    '_gaussian.npy'))))
            gauss_vinit = np.load(Path(str(Path(
                method_dir, 'vinit_mdtb_' + train_sess + '_' + method + \
                    '_gaussian.npy'))))
            min_idx = np.argmin(gauss_err)
            min_gauss_vinit = gauss_vinit[min_idx]
            U_i, _, _ = get_iparcel_dictlearning(
                tdata, vinit_type=vtype, dict_init=min_gauss_vinit.T, \
                    n_parcels=n_clusters, method=method, alpha=alpha)

        # Save and plot individual parcellations
        np.save(str(Path(method_dir, 'iparcel_mdtb_' + train_sess + \
                        '_mni_sym-k17_' + method + '_vtype-' + vtype + '.npy')),
                U_i)
        plt.figure(figsize=(20, 20))
        plot_multi_flat(
            U_i, atlas_name, grid=(6, 4), cmap='tab20', dtype='prob',
            titles=["subj_{}".format(i+1) for i in range(U_i.shape[0])],  
            fig_path=str(Path(method_dir, 'iparcel_mdtb_' + train_sess + \
                            '_mni_sym-k17_' + method + '_vtype-' + vtype + \
                            '.png')))
        # plt.show()
