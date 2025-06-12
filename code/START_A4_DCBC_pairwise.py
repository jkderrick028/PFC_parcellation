import os.path, pickle, scipy
import numpy as np
import Functional_Fusion.atlas_map as am
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from py_util_dx.data_utils import get_roi_pacels, get_roi_vtx_from_fs32k, convert_prob_atlas_to_absolute_labels
import DCBC.dcbc as DCBC
from scipy.stats import ttest_ind
from nitools.cifti import surf_from_cifti
import nitools as nt
import SUITPy.flatmap as flatmap
from visualizations import *

"""
This script computes the DCBC for each pair of parcels in PFC on individualized atlas (data only). 

modified: 2025.05.28
"""


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand
strength = 7.0

# defining ROIs
# large_ROI = 'PFC'
large_ROI = 'visual'
# large_ROI = 'somatosensory'
# large_ROI = 'parietal'

bin_width = 1 # mm

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}_{strength}')
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)


included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(large_ROI)

## loading individualized parcellation
PKL_individualized_parcellation = os.path.join(projectPath, 'results', 'START_A3_bayes_parcellation', f'{dataset_name}_{strength}', f'output_{large_ROI}_masked.pkl')
with open(PKL_individualized_parcellation, 'rb') as pf:
    output_indiv = pickle.load(pf)
    U_indiv = output_indiv['Uhat_data']         # data only parcellation
    labels_in_glasser = output_indiv['labels_in_glasser']
    U_indiv_label = convert_prob_atlas_to_absolute_labels(U_indiv, labels_in_glasser, excluded_vtx_inds_LR)
    parcel_names_in_glasser = output_indiv['parcel_names_in_glasser']

parcels_ROI = get_roi_pacels(large_ROI)
n_parcels = len(parcels_ROI)
n_subjects = U_indiv_label.shape[0]

output = dict()
PKL_output = os.path.join(resultsPath, f'{dataset_name}_{strength}_{large_ROI}_output.pkl')

MAT_dist = os.path.join(projectPath, 'code', 'DCBC', 'distanceMatrix', 'distAvrg_sp.mat')
spatialMat = scipy.io.loadmat(MAT_dist)['avrgDs'].toarray()

# loading MDTB dataset
PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_All_ses-s2.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    X_individuals = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])
X_individuals[np.isnan(X_individuals)] = 0

dcbc_pairwise = np.zeros((n_subjects, n_parcels, n_parcels))
pvals_pairwise = np.zeros((n_parcels, n_parcels))

for parI in np.arange(n_parcels-1):
    for parJ in np.arange(parI+1, n_parcels):
        output[f'{parcel_names_in_glasser[parI]}_{parcel_names_in_glasser[parJ]}'] = {}

        dcbc_across_subjects = []
        within_corrs_across_subjects = []
        between_corrs_across_subjects = []
        results = []
        nums_within = []        # counting the number of vertex pairs within a parcel
        nums_between = []       # counting the number of vertex pairs between parcels
        for subjI in np.arange(n_subjects):
            # get all the vertices that are in the ROI list
            [label_L, label_R] = surf_from_cifti(atlas.data_to_cifti(U_indiv_label[subjI].reshape(1, -1)))
            label_L = label_L.flatten().astype(int)
            vertex_label_ROI, vertex_ind_ROI = [], []
            for i, label in enumerate(label_L):
                if label in [labels_in_glasser[parI], labels_in_glasser[parJ]]:
                    vertex_label_ROI.append(label)
                    vertex_ind_ROI.append(i)

            vertex_label_ROI = np.array(vertex_label_ROI)
            vertex_ind_ROI = np.array(vertex_ind_ROI)

            spaMat = spatialMat[vertex_ind_ROI, :]
            spaMat = spaMat[:, vertex_ind_ROI]

            # only keep the left hemisphere cortex
            select_inds = [x for x in included_vtx_inds_L if U_indiv_label[subjI, x] in [labels_in_glasser[parI], labels_in_glasser[parJ]]]
            data = X_individuals[subjI, :, select_inds]

            myDCBC = DCBC.compute_DCBC(maxDist=35, binWidth=bin_width, parcellation=vertex_label_ROI, func=data,
                                       dist=spaMat, weighting=True, backend='numpy')
            dcbc_across_subjects.append(myDCBC['DCBC'])
            within_corrs_across_subjects.append(myDCBC['corr_within'])
            between_corrs_across_subjects.append(myDCBC['corr_between'])
            results.append(myDCBC)
            nums_within.append(myDCBC['num_within'])
            nums_between.append(myDCBC['num_between'])

        within_corrs = np.array(within_corrs_across_subjects)
        between_corrs = np.array(between_corrs_across_subjects)
        within_corrs_mean = np.nanmean(within_corrs, axis=0)
        within_corrs_ste = np.nanstd(within_corrs, axis=0) / np.sqrt(n_subjects)
        between_corrs_mean = np.nanmean(between_corrs, axis=0)
        between_corrs_ste = np.nanstd(between_corrs, axis=0) / np.sqrt(n_subjects)

        ## testing weather dcbc is significantly higher than 0
        dcbc = np.array(dcbc_across_subjects)
        ttest_result = ttest_ind(dcbc, 0, alternative='greater')
        significance_level = 0.05
        is_significant = ttest_result.pvalue < significance_level
        dcbc_pairwise[:, parI, parJ] = dcbc
        pvals_pairwise[parI, parJ] = ttest_result.pvalue

        fig, ax = plt.subplots(1, 1)
        ax.errorbar(np.arange(0, 35, step=bin_width), within_corrs_mean, yerr=within_corrs_ste)
        ax.errorbar(np.arange(0, 35, step=bin_width), between_corrs_mean, yerr=between_corrs_ste)
        ax.set_xlabel('distance (mm)')
        ax.set_ylabel('vertex-to-vertex correlation')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(['within', 'between'], frameon=False)
        ax.set_title(f'{parcel_names_in_glasser[parI]} {parcel_names_in_glasser[parJ]}, p={ttest_result.pvalue}')

        JPG_fig = os.path.join(resultsPath, f'{parcel_names_in_glasser[parI]}_{parcel_names_in_glasser[parJ]}_dcbc.jpg')
        plt.savefig(JPG_fig, dpi=500, format='jpg')

        fig, ax = plt.subplots(1, 1)
        # ax.bar(np.arange(0, 35, 5), results[0]['num_within'])
        # ax.bar(np.arange(0, 35, 5)+1, results[0]['num_between'])
        ax.bar(np.arange(0, 35, bin_width), np.array(nums_within).mean(axis=0))
        ax.bar(np.arange(0, 35, bin_width) + 0.5, np.array(nums_between).mean(axis=0))

        ax.set_xlabel('distance (mm)')
        ax.set_ylabel('vertex pair counts')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(['within', 'between'], frameon=False)
        ax.set_title(f'{parcel_names_in_glasser[parI]} {parcel_names_in_glasser[parJ]}')

        JPG_fig = os.path.join(resultsPath, f'{parcel_names_in_glasser[parI]}_{parcel_names_in_glasser[parJ]}_counts.jpg')
        plt.savefig(JPG_fig, dpi=500, format='jpg')

        # output[f'{indices_ROI[0]}_{indices_ROI[1]}']['dcbc_results'] = results
        output[f'{parcel_names_in_glasser[parI]}_{parcel_names_in_glasser[parJ]}']['within_corrs'] = within_corrs
        output[f'{parcel_names_in_glasser[parI]}_{parcel_names_in_glasser[parJ]}']['between_corrs'] = between_corrs
        output[f'{parcel_names_in_glasser[parI]}_{parcel_names_in_glasser[parJ]}']['pvalue'] = ttest_result.pvalue
        output[f'{parcel_names_in_glasser[parI]}_{parcel_names_in_glasser[parJ]}']['DCBC'] = dcbc

        plt.close('all')


output['dcbc_pairwise'] = dcbc_pairwise
output['pvals_pairwise'] = pvals_pairwise

with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)
