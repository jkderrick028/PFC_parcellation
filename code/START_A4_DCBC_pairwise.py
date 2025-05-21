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
This script computes the DCBC for each pair of parcels in PFC on glasser group atlas. 

modified: 2025.05.20
"""


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

# load cortical parcellation from label.gii file
glasser_L = os.path.join(surface_helpers_dir, 'glasser.L.label.gii')
glasser_R = os.path.join(surface_helpers_dir, 'glasser.R.label.gii')

gii_file = nib.load(glasser_L)
# gii_file = nib.load(glasser_R)
parcels_inds = gii_file.darrays[0].data

dict_parcel_indices = get_glasser_labels()

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

# defining ROIs
large_ROI = 'PFC'
# large_ROI = 'visual'
# large_ROI = 'somatosensory'
# large_ROI = 'parietal'

output = dict()
PKL_output = os.path.join(resultsPath, f'{dataset_name}_{large_ROI}_output.pkl')

MAT_dist = os.path.join(projectPath, 'code', 'DCBC', 'distanceMatrix', 'distAvrg_sp.mat')
spatialMat = scipy.io.loadmat(MAT_dist)['avrgDs']

# loading MDTB dataset
PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_All_ses-s2.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    X_individuals = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])
X_individuals[np.isnan(X_individuals)] = 0

parcels_ROI = get_roi_pacels(large_ROI)
n_parcels = len(parcels_ROI)

for parI in np.arange(n_parcels-1):
    for parJ in np.arange(parI+1, n_parcels):
        parcel_pair = [parcels_ROI[parI], parcels_ROI[parJ]]
        indices_ROI = [dict_parcel_indices[k] for k in parcel_pair]

        output[f'{parcel_pair[0]}_{parcel_pair[1]}'] = {}

        # get all the vertices that are in the ROI list
        vertex_label_ROI, vertex_ind_ROI = [], []
        for i, label in enumerate(parcels_inds):
            if label in indices_ROI:
                vertex_label_ROI.append(label)
                vertex_ind_ROI.append(i)

        vertex_label_ROI = np.array(vertex_label_ROI)
        vertex_ind_ROI = np.array(vertex_ind_ROI)

        spaMat = spatialMat[vertex_ind_ROI, :]
        spaMat = spaMat[:, vertex_ind_ROI]

        included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(parcel_pair)
        # only keep the left hemisphere cortex
        data = X_individuals[:, :, included_vtx_inds_L]
        n_subjects, n_conditions, n_vertices = data.shape

        # Create a DCBC evaluation object of the desired evaluation parameters(left hemisphere)
        results = []
        within_corrs = []
        between_corrs = []
        dcbc = []

        for subjI in np.arange(n_subjects):
            myDCBC = DCBC.compute_DCBC(maxDist=35, binWidth=5, parcellation=vertex_label_ROI, func=data[subjI].T,
                                       dist=spaMat, weighting=True, backend='numpy')
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
        ax.errorbar(np.arange(0, 35, step=5), within_corrs_mean, yerr=within_corrs_ste)
        ax.errorbar(np.arange(0, 35, step=5), between_corrs_mean, yerr=between_corrs_ste)
        ax.set_xlabel('distance (mm)')
        ax.set_ylabel('vertex-to-vertex correlation')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(['within', 'between'], frameon=False)
        ax.set_title(f'{parcel_pair[0]} {parcel_pair[1]}, p={ttest_result.pvalue}')

        JPG_fig = os.path.join(resultsPath, f'{parcel_pair[0]}_{parcel_pair[1]}_dcbc.jpg')
        plt.savefig(JPG_fig, dpi=500, format='jpg')

        fig, ax = plt.subplots(1, 1)
        ax.bar(np.arange(0, 35, 5), dcbc[0]['num_within'])
        ax.bar(np.arange(0, 35, 5), dcbc[0]['num_between'])
        ax.set_xlabel('distance (mm)')
        ax.set_ylabel('vertex pair counts')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(['within', 'between'], frameon=False)
        ax.set_title(f'{parcel_pair[0]} {parcel_pair[1]}')

        JPG_fig = os.path.join(resultsPath, f'{parcel_pair[0]}_{parcel_pair[1]}_counts.jpg')
        plt.savefig(JPG_fig, dpi=500, format='jpg')

        output[f'{parcel_pair[0]}_{parcel_pair[1]}']['dcbc_results'] = results
        output[f'{parcel_pair[0]}_{parcel_pair[1]}']['within_corrs'] = within_corrs
        output[f'{parcel_pair[0]}_{parcel_pair[1]}']['between_corrs'] = between_corrs
        output[f'{parcel_pair[0]}_{parcel_pair[1]}']['ttest_result'] = ttest_result
        output[f'{parcel_pair[0]}_{parcel_pair[1]}']['DCBC'] = dcbc


with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)
