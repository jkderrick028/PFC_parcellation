import numpy as np
import matplotlib.pyplot as plt
import os, pickle, subprocess
from py_util_dx.py_utils import setProjectPath
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import nibabel as nib
from Functional_Fusion.dataset import decompose_pattern_into_group_indiv_noise
from flat2data import flat2ndarray
import pandas as pd


def mean_centre_predictors(X):
    # subtract out the mean across conditions
    #
    # X (K x P): K (number of conditions), P (number of voxels)

    K, P = X.shape
    result = (np.eye(K) - 1 / K * np.ones(K)) @ X

    return result


def orthogonalize_factors(A, B):
    # orthogonalizing A and B

    B = B - A @ np.linalg.pinv(A) @ B
    return B


def get_predicted(X, Y):
    # the predicted by regression model

    predicted = X @ np.linalg.pinv(X.T @ X) @ X.T @ Y
    return predicted


plt.close('all')
projectPath, mainResultsPath = setProjectPath()
# base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''))
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

smoothing_resultsPath = os.path.join(mainResultsPath, 'exploration8_Demand_smooth_work')

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

smoothing_kernels = ['orig', '10_mm', '8_mm', '6_mm', '4_mm', '2_mm', 'residuals']      # mm, fwhm

# Read data from two gifti files for left and right hemisphere
glasser_left = os.path.join(surface_helpers_dir, 'glasser.L.label.gii')
glasser_right = os.path.join(surface_helpers_dir, 'glasser.R.label.gii')

flat_surf_L = os.path.join(surface_helpers_dir, 'tpl-fs32k_hemi-L_flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'tpl-fs32k_hemi-R_flat.surf.gii')

parcel_dict = {
    'DLPFC': ['9a', '9-46d', '9p', 'SFL', '8BL', 's6-8', '8Ad', 'i6-8', 'a9-46v', '46', '8Av', 'p9-46v', '8C'],
    'somatosensory': ['4', '3a', '3b', '1', '2'],
    'parietal': ['7AL', '7Am', '7Pm', '7PL', 'MIP', 'VIP', '7PC', 'LIPv', 'AIP', 'LIPd'],
    'visual': ['V1', 'V2', 'V3', 'V4']
}

ROI = 'DLPFC'
# ROI = 'visual'
# ROI = 'somatosensory'
# ROI = 'parietal'

output = dict()
PKL_output = os.path.join(resultsPath, f'output_{ROI}.pkl')

included_vtx_inds_LR_dict = {}

for region in parcel_dict.keys():
    included_vtx_inds_LR_dict[region] = []

for region in parcel_dict.keys():
    gii_files = []

    for hemi in ['L', 'R']:
        if hemi == 'L':
            glasser_label = glasser_left
            flat_shape = os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii')
        else:
            glasser_label = glasser_right
            flat_shape = os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii')

        meta = nib.load(flat_shape).meta
        out_gii_file = os.path.join(resultsPath, f'roi_{hemi}_{region}.func.gii')
        roi_data = []
        for roi in parcel_dict[region]:
            hemi_roi = f'{hemi}_{roi}'
            out_label = os.path.join(resultsPath, f'{hemi_roi}.func.gii')
            wb_cmd = f'wb_command -gifti-label-to-roi {glasser_label} {out_label} -name {hemi_roi}_ROI'
            subprocess.run(wb_cmd, shell=True)
            roi_data.append(nib.load(out_label).agg_data())
        roi_data = np.array(roi_data).sum(axis=0)
        out_data = nib.gifti.gifti.GiftiImage(meta=meta)
        out_data.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data=roi_data))
        nib.save(out_data, out_gii_file)
        gii_files.append(out_gii_file)

    label_vec, labels = atlas.get_parcel(gii_files)

    # only keep vertices that are within selected ROIs
    included_vtx_inds_LR = np.where(label_vec > 0)[0]
    included_vtx_inds_L = np.where(label_vec == 1)[0]
    included_vtx_inds_R = np.where(label_vec == 2)[0]
    excluded_vtx_inds_LR = np.where(label_vec == 0)[0]

    included_vtx_inds_LR_dict[region] = included_vtx_inds_LR


# mm.indicator(task)
factor_task = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1]])

factor_difficulty = np.array([[1, 0],
                              [1, 0],
                              [0, 1],
                              [0, 1],
                              [1, 0],
                              [1, 0],
                              [0, 1],
                              [0, 1],
                              [1, 0],
                              [1, 0],
                              [0, 1],
                              [0, 1]])

factor_stimuli = np.array([[1, 0],
                           [0, 1],
                           [1, 0],
                           [0, 1],
                           [1, 0],
                           [0, 1],
                           [1, 0],
                           [0, 1],
                           [1, 0],
                           [0, 1],
                           [1, 0],
                           [0, 1]])

factor_mean = np.ones((12, 1))

# Use kron 
factor_interactions = []
# 2-way interactions between task and difficulty
for i in np.arange(factor_task.shape[1]):
    for j in np.arange(factor_difficulty.shape[1]):
        factor_interactions.append(np.multiply(factor_task[:, i], factor_difficulty[:, j]))

# 2-way interactions between task and stimuli
for i in np.arange(factor_task.shape[1]):
    for k in np.arange(factor_stimuli.shape[1]):
        factor_interactions.append(np.multiply(factor_task[:, i], factor_stimuli[:, k]))

# 2-way interactions between difficulty and stimuli
for j in np.arange(factor_difficulty.shape[1]):
    for k in np.arange(factor_stimuli.shape[1]):
        factor_interactions.append(np.multiply(factor_difficulty[:, j], factor_stimuli[:, k]))

# 3-way interactions among task, difficulty and stimuli
for i in np.arange(factor_task.shape[1]):
    for j in np.arange(factor_difficulty.shape[1]):
        for k in np.arange(factor_stimuli.shape[1]):
            factor_interactions.append(np.multiply(
                np.multiply(factor_task[:, i], factor_difficulty[:, j]),
                factor_stimuli[:, k]))

factor_interactions = np.array(factor_interactions).T

# STEP 1:   mean-centering tasks, difficulty, stimuli
factor_task = mean_centre_predictors(factor_task.T).T
factor_difficulty = mean_centre_predictors(factor_difficulty.T).T
factor_stimuli = mean_centre_predictors(factor_stimuli.T).T


# STEP 2:   orthogonalizing factors then mean-centering
factor_interactions = orthogonalize_factors(np.c_[factor_mean, factor_task, factor_difficulty, factor_stimuli], factor_interactions)
factor_interactions = mean_centre_predictors(factor_interactions.T).T

# check if the factors are orthogonalized now
X = np.c_[factor_mean, factor_task, factor_difficulty, factor_stimuli, factor_interactions]
print(np.linalg.matrix_rank(X))

temp = X.T @ X
print(temp)


PKL_smoothed = os.path.join(smoothing_resultsPath, 'smoothed_orig_mm.pkl')

with open(PKL_smoothed, 'rb') as pf:
    temp = pickle.load(pf)
    X_individuals = temp['X_individuals']
    info_individuals = temp['info_individuals']
    dataset_obj_individuals = temp['dataset_obj_individuals']

X_individuals[np.where(np.isnan(X_individuals))] = 0

task_conds = list(info_individuals.names)

part_vec = list(info_individuals[dataset_obj_individuals.part_ind])
cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

n_subjects = X_individuals.shape[0]
n_partitions = np.unique(part_vec).size
n_conditions = np.unique(cond_vec).size

criterion = 'global'
features = ['original', 'mean', 'task', 'difficulty', 'stimuli', 'interactions']

variances = {}
vs_over_vs_plus_vg = {}
vs_plus_vg = {}

for smoothing_kernel in smoothing_kernels:
    variances[smoothing_kernel] = {}
    for feature in features:
        variances[smoothing_kernel][feature] = []

    vs_over_vs_plus_vg[smoothing_kernel] = []
    vs_plus_vg[smoothing_kernel] = []


n_subsamplings = 20
n_subsamples = 10       # randomly sample 10 out of 37 participants
np.random.seed(0)

subject_indices_presampled = []
for samplingI in np.arange(n_subsamplings):
    subject_indices = np.random.choice(n_subjects, n_subsamples, replace=False)
    subject_indices_presampled.append(subject_indices)

for smoothing_kernel in smoothing_kernels:
    if smoothing_kernel == 'orig':
        X_individuals_smoothed = np.copy(X_individuals)
    else:
        if smoothing_kernel == 'residuals':
            PKL_smoothed = os.path.join(smoothing_resultsPath, 'smoothed_' + smoothing_kernel + '_mm.pkl')
        else:
            PKL_smoothed = os.path.join(smoothing_resultsPath, 'smoothed_'+smoothing_kernel+'.pkl')
        with open(PKL_smoothed, 'rb') as pf:
            X_individuals_smoothed = pickle.load(pf)

    X_individuals_smoothed = np.copy(X_individuals_smoothed[:, :, included_vtx_inds_LR_dict[ROI]])
    n_vertices = X_individuals_smoothed.shape[-1]

    for samplingI in np.arange(n_subsamplings):
        subject_indices = subject_indices_presampled[samplingI]
        data = flat2ndarray(X_individuals_smoothed[subject_indices], part_vec, cond_vec)

        variances[smoothing_kernel]['original'].append(decompose_pattern_into_group_indiv_noise(data, criterion=criterion))     # vg, vs, ve
        data_current = np.zeros(data.shape)

        # STEP 4:   variance decomposition on the mean across conditions
        for subjectI in np.arange(n_subsamples):
            for partI in np.arange(n_partitions):
                data_current[subjectI, partI] = get_predicted(factor_mean, data[subjectI, partI])
        variances[smoothing_kernel]['mean'].append(decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion))

        # STEP 5:   variance decomposition on tasks
        for subjectI in np.arange(n_subsamples):
            for partI in np.arange(n_partitions):
                data_current[subjectI, partI] = get_predicted(factor_task, data[subjectI, partI])
        variances[smoothing_kernel]['task'].append(decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion))

        # STEP 6:   variance decomposition on difficulties
        for subjectI in np.arange(n_subsamples):
            for partI in np.arange(n_partitions):
                data_current[subjectI, partI] = get_predicted(factor_difficulty, data[subjectI, partI])
        variances[smoothing_kernel]['difficulty'].append(decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion))

        # STEP 7:   variance decomposition on stimuli
        for subjectI in np.arange(n_subsamples):
            for partI in np.arange(n_partitions):
                data_current[subjectI, partI] = get_predicted(factor_stimuli, data[subjectI, partI])
        variances[smoothing_kernel]['stimuli'].append(decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion))

        # STEP 8:   variance decomposition on interactions
        for subjectI in np.arange(n_subsamples):
            for partI in np.arange(n_partitions):
                data_current[subjectI, partI] = get_predicted(factor_interactions, data[subjectI, partI])
        variances[smoothing_kernel]['interactions'].append(decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion))


# For each factor, make a table n_smoothing x n_subsamplings x (vg, vs, ve)
var_decomposition_dict = {}
for feature in features:
    tbl = []
    for smoothing_kernel in smoothing_kernels:
        tbl.append(np.array(variances[smoothing_kernel][feature]).squeeze())
    var_decomposition_dict[feature] = np.array(tbl)     # n_smoothing x n_subsamplings x (vg, vs, ve)

output['var_decomposition_dict'] = var_decomposition_dict

# do vg, vs, ve decomposition for original unsmoothed data
data_original_orig = flat2ndarray(X_individuals[:, :, included_vtx_inds_LR_dict[ROI]], part_vec, cond_vec)
variances_original_orig = decompose_pattern_into_group_indiv_noise(data_original_orig, criterion=criterion)

# vs+vg for original, unsmoothed data
orig_vg_plus_vs = np.sum(variances_original_orig[0, 0:2])

plt.figure()
for feature in features:
    tbl = var_decomposition_dict[feature]   # n_smoothing x n_subsamplings x (vg, vs, ve)
    smoothed_vg_plus_vs_over_original_vg_plus_vs = (tbl[:, :, 0] + tbl[:, :, 1]).squeeze() / orig_vg_plus_vs
    smoothed_vg_plus_vs_over_original_vg_plus_vs_mean = smoothed_vg_plus_vs_over_original_vg_plus_vs.mean(axis=1)
    smoothed_vg_plus_vs_over_original_vg_plus_vs_std = smoothed_vg_plus_vs_over_original_vg_plus_vs.std(axis=1)
    smoothed_vg_plus_vs_over_original_vg_plus_vs_ste = smoothed_vg_plus_vs_over_original_vg_plus_vs_std / np.sqrt(n_subsamplings)

    p = plt.scatter(0, smoothed_vg_plus_vs_over_original_vg_plus_vs_mean[0])
    c = p.get_facecolor()[0]
    plt.errorbar(0, smoothed_vg_plus_vs_over_original_vg_plus_vs_mean[0], smoothed_vg_plus_vs_over_original_vg_plus_vs_ste[0], ecolor=c)
    plt.errorbar(np.arange(1, len(smoothing_kernels)), smoothed_vg_plus_vs_over_original_vg_plus_vs_mean[1:], smoothed_vg_plus_vs_over_original_vg_plus_vs_ste[1:], color=c, ecolor=c, label=feature)

plt.xticks(np.arange(len(smoothing_kernels)), labels=smoothing_kernels)
plt.ylabel('vs+vg')
plt.xlabel('smoothing')
plt.title(f'vs+vg normalized by orig vs+vg {ROI}')
plt.legend(frameon=False)
# plt.ylim([0, 1])
# plt.yticks(np.arange(0, 1, step=0.2))
plt.savefig(os.path.join(resultsPath, f'vs_plus_vg_norm_by_vs_plus_vg_{ROI}.png'), dpi=500)


# vs over vs+vg
plt.figure()
for feature in features:
    tbl = var_decomposition_dict[feature]
    vs_over_vs_plus_vg = np.divide(tbl[:, :, 1].squeeze(), (tbl[:, :, 0]+tbl[:, :, 1]).squeeze())

    vs_over_vs_plus_vg_mean = vs_over_vs_plus_vg.mean(axis=1)
    vs_over_vs_plus_vg_std = vs_over_vs_plus_vg.std(axis=1)
    vs_over_vs_plus_vg_ste = vs_over_vs_plus_vg_std / np.sqrt(n_subsamplings)

    p = plt.scatter(0, vs_over_vs_plus_vg_mean[0])
    c = p.get_facecolor()[0]
    plt.errorbar(0, vs_over_vs_plus_vg_mean[0], vs_over_vs_plus_vg_ste[0], ecolor=c)
    plt.errorbar(np.arange(1, len(smoothing_kernels)), vs_over_vs_plus_vg_mean[1:], vs_over_vs_plus_vg_ste[1:], ecolor=c, color=c, label=feature)

plt.xticks(np.arange(len(smoothing_kernels)), labels=smoothing_kernels)
plt.ylabel('vs/vs+vg')
plt.xlabel('smoothing')
plt.title(f'vs/vs+vg on each factor {ROI}')
plt.legend(frameon=False)
# plt.ylim([0, 1])
plt.yticks(np.arange(0, 1, step=0.2))
plt.savefig(os.path.join(resultsPath, f'vs_over_vs_plus_vg_{ROI}.png'), dpi=500)


# vs+vg over vs+vg+ve
plt.figure()
for feature in features:
    tbl = var_decomposition_dict[feature]
    vs_plus_vg_over_vs_vg_ve = np.divide((tbl[:, :, 0] + tbl[:, :, 1]).squeeze(), (tbl[:, :, 0] + tbl[:, :, 1] + tbl[:, :, 2]).squeeze())

    vs_plus_vg_over_vs_vg_ve_mean = vs_plus_vg_over_vs_vg_ve.mean(axis=1)
    vs_plus_vg_over_vs_vg_ve_std = vs_plus_vg_over_vs_vg_ve.std(axis=1)
    vs_plus_vg_over_vs_vg_ve_ste = vs_plus_vg_over_vs_vg_ve_std / np.sqrt(n_subsamplings)

    p = plt.scatter(0, vs_plus_vg_over_vs_vg_ve_mean[0])
    c = p.get_facecolor()[0]
    plt.errorbar(0, vs_plus_vg_over_vs_vg_ve_mean[0], vs_plus_vg_over_vs_vg_ve_ste[0], ecolor=c)
    plt.errorbar(np.arange(1, len(smoothing_kernels)), vs_plus_vg_over_vs_vg_ve_mean[1:], vs_plus_vg_over_vs_vg_ve_ste[1:],
                 ecolor=c, color=c, label=feature)

plt.xticks(np.arange(len(smoothing_kernels)), labels=smoothing_kernels)
plt.ylabel('vs+vg/vs+vg+ve')
plt.xlabel('smoothing')
plt.title(f'vs+vg/vs+vg+ve on each factor {ROI}')
plt.legend(frameon=False)
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1, step=0.2))
plt.savefig(os.path.join(resultsPath, f'vs_plus_vg_over_vs_vg_ve_{ROI}.png'), dpi=500)

plt.show()

output['variances'] = variances
with open(PKL_output, 'wb') as pf:
    pickle.dump(output, pf)

plt.close('all')
