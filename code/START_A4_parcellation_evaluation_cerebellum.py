import os.path, pickle
import Functional_Fusion.atlas_map as am
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from scipy.stats import ttest_ind
from evaluations import *
import DCBC


"""
This script computes DCBC using individualized parcellation for cerebellum. 

modified: 2025.05.21
"""


## defining paths
projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

appendix = 'cerebellum'

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

output = dict()
PKL_output = os.path.join(resultsPath, f'{dataset_name}_{appendix}_output.pkl')

## loading MDTB data
PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_All_ses-s2_MNISymC3.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    data = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

# fill nans with 0
data[np.isnan(data)] = 0
n_subjects = data.shape[0]

PKL_individualized_parcellation = os.path.join(projectPath, 'results', 'START_A3_bayes_parcellation_cerebellum', dataset_name, f'output_{appendix}.pkl')
with open(PKL_individualized_parcellation, 'rb') as pf:
    output_indiv = pickle.load(pf)
    U_indiv = output_indiv['Uhat_data']
    U_indiv_label = np.argmax(U_indiv, axis=1) + 1
    U_group_label = np.argmax(output_indiv['U'], axis=0) + 1

## loading group atlas
atlas, _ = am.get_atlas('MNISymC3')
atlas_fname = os.path.join(surface_helpers_dir, 'atl-NettekovenSym32_space-MNI152NLin2009cSymC_probseg.nii.gz')
U_group = atlas.read_data(atlas_fname)
U_group = np.tile(U_group, (n_subjects, 1, 1))
U_group[np.isnan(U_group)] = 0

## prediction error with leave-one-subject-out cross-validation
cosine_distances_group = prediction_error_cv(U_group, data)
cosine_distances_indiv = prediction_error_cv(U_indiv, data)

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

spatialMat = DCBC.utilities.compute_dist(atlas.world.T, resolution=1)

U_group_label = np.tile(U_group_label, (n_subjects, 1))
output_dcbc_group = compute_dcbc_indiv(U_group_label, data, spatialMat)
output_dcbc_indiv = compute_dcbc_indiv(U_indiv_label, data, spatialMat)

## making results figures for group atlas
within_corrs = output_dcbc_group['within_corrs']
between_corrs = output_dcbc_group['between_corrs']
dcbc_group = output_dcbc_group['dcbc']
JPG_fig = os.path.join(resultsPath, f'DCBC_group_{appendix}.jpg')


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
    ax.set_title(f'{appendix} {atlas_type}, p={ttest_result.pvalue}')

    plt.savefig(fig_path, dpi=500, format='jpg')


summarize_dcbc(within_corrs, between_corrs, dcbc_group, JPG_fig, 'group')

## making results figures for individualized atlas
within_corrs = output_dcbc_indiv['within_corrs']
between_corrs = output_dcbc_indiv['between_corrs']
dcbc_indiv = output_dcbc_indiv['dcbc']
JPG_fig = os.path.join(resultsPath, f'DCBC_indiv_{appendix}.jpg')

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

output['output_dcbc_group'] = output_dcbc_group
output['output_dcbc_indiv'] = output_dcbc_indiv
output['cosine_distances_group'] = cosine_distances_group
output['cosine_distances_indiv'] = cosine_distances_indiv

with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)



