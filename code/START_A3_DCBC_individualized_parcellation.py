import os.path, pickle, subprocess, scipy
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

import Functional_Fusion.atlas_map as am
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from py_util_dx.data_utils import get_roi_pacels, get_glasser_labels, get_roi_vtx_from_fs32k
import DCBC.dcbc as DCBC
from scipy.stats import ttest_ind
from evaluations import *


"""
This script computes DCBC using individualized parcellation for PFC. 

modified: 2025.05.05
"""


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

## loading individualized parcellation
included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k('PFC')

PKL_individualized_parcellation = os.path.join(projectPath, 'results', 'START_A5_bayes_parcellation', dataset_name, 'output_PFC_masked.pkl')
with open(PKL_individualized_parcellation, 'rb') as pf:
    output_indiv = pickle.load(pf)
    U_indiv_label = np.argmax(output_indiv['U_indiv'], axis=1) + 1

U_indiv_label = U_indiv_label[:, included_vtx_inds_L]

dict_parcel_indices = get_glasser_labels()

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

# defining ROIs
large_ROI = 'PFC'
# large_ROI = 'visual'
# large_ROI = 'somatosensory'
# large_ROI = 'parietal'

MAT_dist = os.path.join(projectPath, 'code', 'DCBC', 'distanceMatrix', 'distAvrg_sp.mat')
spatialMat = scipy.io.loadmat(MAT_dist)['avrgDs']

glasser_L = os.path.join(surface_helpers_dir, 'glasser.L.label.gii')
glasser_R = os.path.join(surface_helpers_dir, 'glasser.R.label.gii')

# extract the first data array as the parcels
# make sure that the input parcels are of shape (N,)
gii_file = nib.load(glasser_L)
# gii_file = nib.load(glasser_R)
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

# loading MDTB dataset
PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_All_ses-s2.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    X_individuals = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])
# fill nans with 0
X_individuals[np.isnan(X_individuals)] = 0
# whole cortex
data = X_individuals[:, :, included_vtx_inds_L]

n_subjects, n_conditions, n_vertices = data.shape

## prediction error
cosine_distances = prediction_error_cv(output_indiv['U_indiv'][:, :, included_vtx_inds_LR], X_individuals[:, :, included_vtx_inds_LR])

# Create a DCBC evaluation object of the desired evaluation parameters(left hemisphere)
results = []
within_corrs = []
between_corrs = []
dcbc = []

output = dict()
PKL_output = os.path.join(resultsPath, f'{dataset_name}_{large_ROI}_output.pkl')

for subjI in np.arange(n_subjects):
    myDCBC = DCBC.compute_DCBC(maxDist=35, binWidth=5, parcellation=U_indiv_label[subjI], func=data[subjI].T, dist=spatialMat, weighting=True, backend='numpy')
    results.append(myDCBC)
    within_corrs.append(myDCBC['corr_within'])
    between_corrs.append(myDCBC['corr_between'])
    dcbc.append(myDCBC['DCBC'])

within_corrs = np.array(within_corrs)
between_corrs = np.array(between_corrs)
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
ax.errorbar(np.arange(0, 35, step=5), within_corrs_mean, yerr = within_corrs_ste)
ax.errorbar(np.arange(0, 35, step=5), between_corrs_mean, yerr=between_corrs_ste)
ax.set_xlabel('distance (mm)')
ax.set_ylabel('vertex-to-vertex correlation')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(['within', 'between'], frameon=False)
if is_significant:
    ax.set_title(f'{large_ROI} individualized, dcbc significant')
else:
    ax.set_title(f'{large_ROI} individualized, dcbc not significant')

JPG_fig = os.path.join(resultsPath, f'DCBC_{large_ROI}.jpg')
plt.savefig(JPG_fig, dpi=500, format='jpg')

output['dcbc_results'] = results
output['within_corrs'] = within_corrs
output['between_corrs'] = between_corrs
output['ttest_result'] = ttest_result
output['DCBC'] = dcbc

with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)
