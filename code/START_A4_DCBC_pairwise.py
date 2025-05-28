import os.path, pickle, scipy
import numpy as np
import Functional_Fusion.atlas_map as am
import nibabel as nib
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from py_util_dx.data_utils import get_roi_pacels, get_glasser_labels, get_roi_vtx_from_fs32k
import DCBC.dcbc as DCBC
from scipy.stats import ttest_ind


"""
This script computes the DCBC for each pair of parcels in PFC on individualized atlas (data only). 

modified: 2025.05.28
"""


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand
strength = 7.0

# defining ROIs
large_ROI = 'PFC'
# large_ROI = 'visual'
# large_ROI = 'somatosensory'
# large_ROI = 'parietal'

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}_{strength}')
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)


## loading individualized parcellation
PKL_individualized_parcellation = os.path.join(projectPath, 'results', 'START_A3_bayes_parcellation', f'{dataset_name}_{strength}', f'output_{large_ROI}_masked.pkl')
with open(PKL_individualized_parcellation, 'rb') as pf:
    output_indiv = pickle.load(pf)
    U_indiv = output_indiv['Uhat_data']         # data only parcellation
    U_indiv_label = np.argmax(U_indiv, axis=1) + 1


## convert the original glasser parcellation label array into individualized label array. 0 for vertices outside of ROI
# load cortical parcellation from label.gii file
glasser_L = os.path.join(surface_helpers_dir, 'glasser.L.label.gii')
glasser_R = os.path.join(surface_helpers_dir, 'glasser.R.label.gii')

gii_file = nib.load(glasser_L)
# gii_file = nib.load(glasser_R)
parcels_inds = gii_file.darrays[0].data

dict_parcel_indices = get_glasser_labels()

included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(large_ROI)

parcels_ROI = get_roi_pacels(large_ROI)
n_parcels = len(parcels_ROI)

indices_ROI = [dict_parcel_indices[p] for p in parcels_ROI]

# get all the vertices that are in the ROI list
vertex_label_ROI, vertex_ind_ROI = [], []
for i, label in enumerate(parcels_inds):
    if label in indices_ROI:
        vertex_label_ROI.append(label)
        vertex_ind_ROI.append(i)

n_subjects = U_indiv_label.shape[0]
parcels_inds_indiv = np.zeros((n_subjects, len(parcels_inds)))
parcels_inds_indiv[:, vertex_ind_ROI] = U_indiv_label[:, included_vtx_inds_L]

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

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

for parI in np.arange(n_parcels-1):
    for parJ in np.arange(parI+1, n_parcels):
        indices_ROI = [parI+1, parJ+1]
        output[f'{indices_ROI[0]}_{indices_ROI[1]}'] = {}

        dcbc_across_subjects = []
        within_corrs_across_subjects = []
        between_corrs_across_subjects = []
        results = []
        for subjI in np.arange(n_subjects):
            # get all the vertices that are in the ROI list
            vertex_label_ROI, vertex_ind_ROI = [], []
            for i, label in enumerate(parcels_inds_indiv[subjI]):
                if label in indices_ROI:
                    vertex_label_ROI.append(label)
                    vertex_ind_ROI.append(i)

            vertex_label_ROI = np.array(vertex_label_ROI)
            vertex_ind_ROI = np.array(vertex_ind_ROI)

            spaMat = spatialMat[vertex_ind_ROI, :]
            spaMat = spaMat[:, vertex_ind_ROI]

            # only keep the left hemisphere cortex
            select_inds = [x for x in np.arange(len(parcels_inds_indiv[subjI])) if parcels_inds_indiv[subjI][x] in indices_ROI]
            data = X_individuals[subjI, :, select_inds]

            myDCBC = DCBC.compute_DCBC(maxDist=35, binWidth=5, parcellation=vertex_label_ROI, func=data,
                                       dist=spaMat, weighting=True, backend='numpy')
            dcbc_across_subjects.append(myDCBC['DCBC'])
            within_corrs_across_subjects.append(myDCBC['corr_within'])
            between_corrs_across_subjects.append(myDCBC['corr_between'])
            results.append(myDCBC)

        within_corrs = np.array(within_corrs_across_subjects)
        between_corrs = np.array(between_corrs_across_subjects)
        within_corrs_mean = np.mean(within_corrs, axis=0)
        within_corrs_ste = np.std(within_corrs, axis=0) / np.sqrt(n_subjects)
        between_corrs_mean = np.mean(between_corrs, axis=0)
        between_corrs_ste = np.std(between_corrs, axis=0) / np.sqrt(n_subjects)

        ## testing weather dcbc is significantly higher than 0
        dcbc = np.array(dcbc_across_subjects)
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
        ax.set_title(f'{indices_ROI[0]} {indices_ROI[1]}, p={ttest_result.pvalue}')

        JPG_fig = os.path.join(resultsPath, f'{indices_ROI[0]}_{indices_ROI[1]}_dcbc.jpg')
        plt.savefig(JPG_fig, dpi=500, format='jpg')

        fig, ax = plt.subplots(1, 1)
        ax.bar(np.arange(0, 35, 5), results[0]['num_within'])
        ax.bar(np.arange(0, 35, 5)+1, results[0]['num_between'])
        ax.set_xlabel('distance (mm)')
        ax.set_ylabel('vertex pair counts')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(['within', 'between'], frameon=False)
        ax.set_title(f'{indices_ROI[0]} {indices_ROI[1]}')

        JPG_fig = os.path.join(resultsPath, f'{indices_ROI[0]}_{indices_ROI[1]}_counts.jpg')
        plt.savefig(JPG_fig, dpi=500, format='jpg')

        output[f'{indices_ROI[0]}_{indices_ROI[1]}']['dcbc_results'] = results
        output[f'{indices_ROI[0]}_{indices_ROI[1]}']['within_corrs'] = within_corrs
        output[f'{indices_ROI[0]}_{indices_ROI[1]}']['between_corrs'] = between_corrs
        output[f'{indices_ROI[0]}_{indices_ROI[1]}']['ttest_result'] = ttest_result
        output[f'{indices_ROI[0]}_{indices_ROI[1]}']['DCBC'] = dcbc

        plt.close('all')


with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)
