import os.path, pickle, scipy, torch
import numpy as np
import HierarchBayesParcel.arrangements as ar
import Functional_Fusion.atlas_map as am
import nibabel as nib
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from py_util_dx.data_utils import get_roi_pacels, get_glasser_labels, get_roi_vtx_from_fs32k
from scipy.stats import ttest_ind
from evaluations import *


"""
This script computes DCBC using individualized parcellation for PFC. 

modified: 2025.05.05
"""


## defining paths
projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

strength = 1.0

# defining ROIs
large_ROI = 'PFC'
# large_ROI = 'visual'
# large_ROI = 'somatosensory'
# large_ROI = 'parietal'

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}_{strength}')
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

output = dict()
PKL_output = os.path.join(resultsPath, f'{dataset_name}_{large_ROI}_output.pkl')

## loading MDTB data
PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_All_ses-s2.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    X_individuals = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])
# fill nans with 0
X_individuals[np.isnan(X_individuals)] = 0
n_subjects = X_individuals.shape[0]

## loading individualized parcellation and glasser group parcellation
included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(large_ROI)

PKL_individualized_parcellation = os.path.join(projectPath, 'results', 'START_A3_bayes_parcellation', dataset_name, f'output_{large_ROI}_masked.pkl')
with open(PKL_individualized_parcellation, 'rb') as pf:
    output_indiv = pickle.load(pf)
    U_indiv = output_indiv['Uhat_data']         # data only parcellation
    U_indiv_label = np.argmax(U_indiv, axis=1) + 1
    U_group_label = output_indiv['U']

# since the U_group_label is 1d, we need to convert it to a parcel x vertex soft probablistic atlas
_, U_group = np.unique(U_group_label, return_inverse=True)
K = np.unique(U_group).size

logpi = ar.expand_mn_1d(U_group, K) * strength
logpi = logpi[1:, :]
U_group = torch.softmax(logpi, dim=0).detach().numpy()
U_group = np.tile(U_group, (n_subjects, 1, 1))

## prediction error with leave-one-subject-out cross-validation
cosine_distances_group = prediction_error_cv(U_group[:, :, included_vtx_inds_LR], X_individuals[:, :, included_vtx_inds_LR])
cosine_distances_indiv = prediction_error_cv(U_indiv[:, :, included_vtx_inds_LR], X_individuals[:, :, included_vtx_inds_LR])

x_group = np.ones(len(cosine_distances_group))
x_indiv = 2*np.ones(len(cosine_distances_indiv))
x_cosine_dist = np.concatenate([x_group, x_indiv])
cosine_dist_concat = np.concatenate([cosine_distances_group, cosine_distances_indiv])
plt.figure()
plt.scatter(x_cosine_dist, cosine_dist_concat)
plt.xticks([1, 2], labels=['group', 'indiv'])
plt.xlabel('atlas type')
plt.ylabel('cosine dist')
plt.title('prediction error using glasser and indiv atlas')
JPG_fig = os.path.join(resultsPath, f'prediction_error_{large_ROI}.jpg')
plt.savefig(JPG_fig, dpi=500, format='jpg')

## DCBC using left hemisphere only
# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

MAT_dist = os.path.join(projectPath, 'code', 'DCBC', 'distanceMatrix', 'distAvrg_sp.mat')
spatialMat = scipy.io.loadmat(MAT_dist)['avrgDs']

glasser_L = os.path.join(surface_helpers_dir, 'glasser.L.label.gii')
glasser_R = os.path.join(surface_helpers_dir, 'glasser.R.label.gii')

# extract the first data array as the parcels
# make sure that the input parcels are of shape (N,)
gii_file = nib.load(glasser_L)

dict_parcel_indices = get_glasser_labels()  # {parcel: label}

parcels_inds = gii_file.darrays[0].data
parcels_ROI = get_roi_pacels(large_ROI)
indices_ROI = [dict_parcel_indices[k] for k in parcels_ROI]

# get all the vertices that are in the ROI list
vertex_label_ROI, vertex_ind_ROI = [], []
for i, label in enumerate(parcels_inds):
    if label in indices_ROI:
        vertex_label_ROI.append(label)
        vertex_ind_ROI.append(i)

vertex_label_ROI = np.array(vertex_label_ROI)
vertex_ind_ROI = np.array(vertex_ind_ROI)

spatialMat = spatialMat[vertex_ind_ROI, :]
spatialMat = spatialMat[:, vertex_ind_ROI]

# Create a DCBC evaluation object of the desired evaluation parameters(left hemisphere)
U_indiv_label = U_indiv_label[:, included_vtx_inds_L]
U_group_label = np.tile(U_group_label, (n_subjects, 1))[:, included_vtx_inds_L]
data = X_individuals[:, :, included_vtx_inds_L]

output_dcbc_group = compute_dcbc_indiv(U_group_label, data, spatialMat)
output_dcbc_indiv = compute_dcbc_indiv(U_indiv_label, data, spatialMat)

## making results figures for group atlas
within_corrs = output_dcbc_group['within_corrs']
between_corrs = output_dcbc_group['between_corrs']
dcbc_group = output_dcbc_group['dcbc']
JPG_fig = os.path.join(resultsPath, f'DCBC_group_{large_ROI}.jpg')


def summarize_dcbc(within_corrs, between_corrs, dcbc, fig_path, atlas_type):
    """
    summarizing dcbc results
    Args:
        within_corrs:
        between_corrs:
        dcbc:

    Returns:

    """

    within_corrs_mean = np.mean(within_corrs, axis=0)
    within_corrs_ste = np.std(within_corrs, axis=0) / np.sqrt(n_subjects)
    between_corrs_mean = np.mean(between_corrs, axis=0)
    between_corrs_ste = np.std(between_corrs, axis=0) / np.sqrt(n_subjects)

    ## testing weather dcbc is significantly higher than 0
    dcbc = np.array(dcbc)
    ttest_result = ttest_ind(dcbc, 0, alternative='greater')
    significance_level = 0.05
    is_significant = ttest_result.pvalue < significance_level

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(np.arange(0, 35, step=5), within_corrs_mean, yerr=within_corrs_ste)
    ax.errorbar(np.arange(0, 35, step=5), between_corrs_mean, yerr=between_corrs_ste)
    ax.set_xlabel('distance (mm)')
    ax.set_ylabel('vertex-to-vertex correlation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(['within', 'between'], frameon=False)
    if is_significant:
        ax.set_title(f'{large_ROI} {atlas_type}, dcbc significant')
    else:
        ax.set_title(f'{large_ROI} {atlas_type}, dcbc not significant')

    plt.savefig(fig_path, dpi=500, format='jpg')


summarize_dcbc(within_corrs, between_corrs, dcbc_group, JPG_fig, 'group')

## making results figures for individualized atlas
within_corrs = output_dcbc_indiv['within_corrs']
between_corrs = output_dcbc_indiv['between_corrs']
dcbc_indiv = output_dcbc_indiv['dcbc']
JPG_fig = os.path.join(resultsPath, f'DCBC_indiv_{large_ROI}.jpg')

summarize_dcbc(within_corrs, between_corrs, dcbc_indiv, JPG_fig, 'indiv')

x_group = np.ones(len(dcbc_group))
x_indiv = 2*np.ones(len(dcbc_indiv))
x_dcbc = np.concatenate([x_group, x_indiv])
dcbc_concat = np.concatenate([dcbc_group, dcbc_indiv])
plt.figure()
plt.scatter(x_dcbc, dcbc_concat)
plt.xticks([1, 2], labels=['group', 'indiv'])
plt.xlabel('atlas type')
plt.ylabel('dcbc')
plt.title('DCBC using glasser and indiv atlas')
JPG_fig = os.path.join(resultsPath, f'DCBC_scatter_{large_ROI}.jpg')
plt.savefig(JPG_fig, dpi=500, format='jpg')

output['output_dcbc_group'] = output_dcbc_group
output['output_dcbc_indiv'] = output_dcbc_indiv
output['cosine_distances_group'] = cosine_distances_group
output['cosine_distances_indiv'] = cosine_distances_indiv

with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)



