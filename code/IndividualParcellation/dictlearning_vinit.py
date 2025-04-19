"""
Estimate dictionary initialization for data-driven models

author: Ana Luisa Pinho
email: agrilopi@uwo.ca

created: January 29, 2024
last update: July 2024

Compatibility: Python 3.11.5
"""

from pathlib import Path
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import Functional_Fusion.dataset as ds
import HierarchBayesParcel.HierarchBayesParcel.full_model as fm

from global_config import MODEL_DIR, BASE_DIR, DEVICE
from utils_dictionary import get_iparcel_dictlearning
from utils import pretrained_arrmodel


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


def plot_histogram(sorted_gerrs, min_xlabel, max_xlabel, n_bins, color, 
                   write_dir, n_iter, method, histlabel, sess, vtype, 
                   kerr=None, hbperr=None):
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(left=.1, right=.98, bottom=.2, wspace=.075)
    bin_edges = np.linspace(min_xlabel, max_xlabel, n_bins).astype('int')
    # Create a histogram with a density estimate line
    sns.histplot(sorted_gerrs, kde=True, bins=bin_edges, label=histlabel, 
                 color=color)
    if kerr is not None:
        ax.plot(kerr, 1., marker='o', markersize=8, color='tab:olive',
                label='K-Means error')
    if hbperr is not None:
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
    # Save figure
    plt.savefig(str(Path(
        write_dir, 'hist_mdtb_' + sess + '_' + method + '_' + vtype + \
            '.png')))


# ######################### INPUTS ######################################

## Load the atlas
atlas_name = 'MNISymC3'
sym_type = 'asym'
model_name = f'/Models_03/asym_Md_space-MNISymC3_K-17'

# Method's parameters
method = 'online'
alpha = .01
n_rsample = 2000
train_sess='ses-s2' # for mdtb: 'all', 'ses-s1' or 'ses-s2'

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

    # # ########### ESTIMATE THE BEST DICTIONARY INITIALIZATION ############

    output_dir = Path(str(Path(main_dir, method)))
    # Create output folder, if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute error using K-means V_init
    _, _, kerrors = zip(*Parallel(n_jobs=5)(
        delayed(get_iparcel_dictlearning)(
            tdata, vinit_type='kmeans', n_parcels=n_clusters, method=method, 
            alpha=alpha, n_iter=rs)
        for rs in np.arange(n_rsample)))

    # Compute error using HBP V_init
    hbp_vinit = get_hbp_v(ar_model, atlas, tdata, tinfo)
    _, _, hbperrors = zip(*Parallel(n_jobs=5)(
        delayed(get_iparcel_dictlearning)(
            tdata, vinit_type='hbp', dict_init=hbp_vinit, n_parcels=n_clusters,
            method=method, alpha=alpha, n_iter=rs)
        for rs in np.arange(n_rsample)))

    # Compute error using gaussian V_init
    _, gaussian_vinits, gerrors = zip(*Parallel(n_jobs=5)(
        delayed(get_iparcel_dictlearning)(
            tdata, vinit_type='gaussian', n_parcels=n_clusters,
            method=method, alpha=alpha, n_iter=rs)
        for rs in np.arange(n_rsample)))

    # # ******************************************************************

    # Save results
    np.save(str(Path(output_dir, 'errs_mdtb_' + train_sess + '_' + \
                        method + '_kmeans.npy')), kerrors)
    np.save(str(Path(output_dir, 'errs_mdtb_' + train_sess + '_' + \
                        method + '_hbp.npy')), hbperrors)
    np.save(str(Path(output_dir, 'errs_mdtb_' + train_sess + '_' + \
                        method + '_gaussian.npy')), gerrors)
    np.save(str(Path(output_dir, 'vinit_mdtb_' + train_sess + '_' + \
                        method + '_gaussian.npy')), gaussian_vinits)

    # *********************** Visualization ******************************

    # Load results
    kerrors = np.load(str(Path(output_dir, 'errs_mdtb_' + train_sess + \
                               '_' + method + '_kmeans.npy')))

    hbperrors = np.load(str(Path(output_dir, 'errs_mdtb_' + train_sess + \
                                 '_' + method + '_hbp.npy')))

    gerrors = np.load(str(Path(output_dir, 'errs_mdtb_' + train_sess + \
                               '_' + method + '_gaussian.npy')))

    # Plot histograms with density estimate line for each random seed
    errors = [kerrors, hbperrors, gerrors]
    colors = ['tab:olive', 'tab:pink', 'tab:blue']
    labels = ['Kmeans error', 'HBP error', 'Gaussian error']
    vtypes = ['kmeans', 'hbp', 'gaussian']
    for error, color, label, vtype in zip(errors, colors, labels, vtypes):
        min_x = np.amin(error)
        max_x = np.amax(error)
        plot_histogram(error, min_x, max_x, 12, color, output_dir, n_rsample, 
                       method, label, train_sess, vtype)
        del error