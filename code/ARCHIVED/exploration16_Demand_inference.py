import numpy as np
import matplotlib.pyplot as plt
import os, pickle, subprocess
from py_util_dx.py_utils import setProjectPath
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import nibabel as nib
from Functional_Fusion.dataset import decompose_pattern_into_group_indiv_noise
from flat2data import flat2ndarray


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


def get_residuals(X, Y):
    # regress out X from Y and return residuals

    K = X.shape[0]
    residual = (np.eye(K) - X @ np.linalg.pinv(X.T @ X) @ X.T) @ Y
    return residual


def get_predicted(X, Y):
    # the predicted by regression model

    predicted = X @ np.linalg.pinv(X.T @ X) @ X.T @ Y
    return predicted


projectPath, mainResultsPath = setProjectPath()
base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')
# base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''))
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

output = dict()
PKL_output = os.path.join(resultsPath, 'output.pkl')

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

smoothing_kernels = [10, 8, 6, 4, 2]      # mm, fwhm

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

included_vtx_inds_LR_dict = {}

for region in parcel_dict.keys():
    included_vtx_inds_LR_dict[region] = []

output['orig'] = {}
for smoothing_kernel in smoothing_kernels:
    output[f'{smoothing_kernel}_mm'] = {}

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

# factor_task = mean_centre_predictors(factor_task)
# factor_difficulty = mean_centre_predictors(factor_difficulty)
# factor_stimuli = mean_centre_predictors(factor_stimuli)


# STEP 2:   orthogonalizing factors then mean-centering
factor_interactions = orthogonalize_factors(np.c_[factor_mean, factor_task, factor_difficulty, factor_stimuli], factor_interactions)
factor_interactions = mean_centre_predictors(factor_interactions.T).T

# factor_interactions = mean_centre_predictors(factor_interactions)


# check if the factors are orthogonalized now
temp = np.c_[factor_mean, factor_task, factor_difficulty, factor_stimuli, factor_interactions]
temp = temp.T @ temp
print(temp)

# STEP 3:   variance decomposition on original data
#           report vs, vg, ve, vs+vg, vs/(vs+vg)
X_individuals, info_individuals, dataset_obj_individuals = ds.get_dataset(base_dir,
                                                                          dataset='Demand',
                                                                          atlas='fs32k',
                                                                          subj=None,
                                                                          sess='all',
                                                                          type='CondHalf')

X_individuals[np.where(np.isnan(X_individuals))] = 0

X_individuals = X_individuals[:, :, included_vtx_inds_LR_dict['DLPFC']]

task_conds = list(info_individuals.names)

part_vec = list(info_individuals[dataset_obj_individuals.part_ind])
cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

n_subjects = X_individuals.shape[0]
n_partitions = np.unique(part_vec).size
n_vertices = X_individuals.shape[-1]

criterion = 'global'

variances = {}
vs_over_vs_plus_vg = {}
vs_plus_vg = {}

variances['orig'] = {}
vs_over_vs_plus_vg['orig'] = []
vs_plus_vg['orig'] = []
for smoothing_kernel in smoothing_kernels:
    variances[f'{smoothing_kernel}_mm'] = {}
    vs_over_vs_plus_vg[f'{smoothing_kernel}_mm'] = []
    vs_plus_vg[f'{smoothing_kernel}_mm'] = []

smoothing_kernels.insert(0, 'orig')

for smoothing_kernel in smoothing_kernels:
    if smoothing_kernel == 'orig':
        X_individuals_smoothed = X_individuals
    else:
        PKL_smoothed = os.path.join(mainResultsPath, 'exploration8_Demand_smooth_work', f'smoothed_{smoothing_kernel}_mm.pkl')
        with open(PKL_smoothed, 'rb') as pf:
            X_individuals_smoothed = pickle.load(pf)
        smoothing_kernel = f'{smoothing_kernel}_mm'
        X_individuals_smoothed = X_individuals_smoothed[:, :, included_vtx_inds_LR_dict['DLPFC']]

    # whole cortex
    data = flat2ndarray(X_individuals_smoothed, part_vec, cond_vec)
    variances[smoothing_kernel]['original'] = decompose_pattern_into_group_indiv_noise(data, criterion=criterion)     # vg, vs, ve

    # STEP 4:   variance decomposition on the mean across conditions
    data_current = data
    for subjectI in np.arange(n_subjects):
        for partI in np.arange(n_partitions):
            data_current[subjectI, partI] = get_predicted(factor_mean, data[subjectI, partI])
    variances[smoothing_kernel]['mean'] = decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion)

    # STEP 5:   variance decomposition on tasks
    for subjectI in np.arange(n_subjects):
        for partI in np.arange(n_partitions):
            data_current[subjectI, partI] = get_predicted(factor_task, data[subjectI, partI])
    variances[smoothing_kernel]['task'] = decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion)

    # STEP 6:   variance decomposition on difficulties
    for subjectI in np.arange(n_subjects):
        for partI in np.arange(n_partitions):
            data_current[subjectI, partI] = get_predicted(factor_difficulty, data[subjectI, partI])
    variances[smoothing_kernel]['difficulty'] = decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion)

    # STEP 7:   variance decomposition on stimuli
    for subjectI in np.arange(n_subjects):
        for partI in np.arange(n_partitions):
            data_current[subjectI, partI] = get_predicted(factor_stimuli, data[subjectI, partI])
    variances[smoothing_kernel]['stimuli'] = decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion)

    # STEP 8:   variance decomposition on interactions
    for subjectI in np.arange(n_subjects):
        for partI in np.arange(n_partitions):
            data_current[subjectI, partI] = get_predicted(factor_interactions, data[subjectI, partI])
    variances[smoothing_kernel]['interactions'] = decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion)

    # STEP 9:   variance decomposition on all the main effects
    # for subjectI in np.arange(n_subjects):
    #     for partI in np.arange(n_partitions):
    #         data_current[subjectI, partI] = get_predicted(np.c_[factor_task, factor_difficulty, factor_stimuli], data[subjectI, partI])
    # variances[smoothing_kernel]['mains'] = decompose_pattern_into_group_indiv_noise(data_current, criterion=criterion)

    vs_over_vs_plus_vg[smoothing_kernel].append(variances[smoothing_kernel]['original'][0, 1] / np.sum(variances[smoothing_kernel]['original'][0, 0:2]))
    vs_over_vs_plus_vg[smoothing_kernel].append(variances[smoothing_kernel]['mean'][0, 1] / np.sum(variances[smoothing_kernel]['mean'][0, 0:2]))
    vs_over_vs_plus_vg[smoothing_kernel].append(variances[smoothing_kernel]['task'][0, 1] / np.sum(variances[smoothing_kernel]['task'][0, 0:2]))
    vs_over_vs_plus_vg[smoothing_kernel].append(variances[smoothing_kernel]['difficulty'][0, 1] / np.sum(variances[smoothing_kernel]['difficulty'][0, 0:2]))
    vs_over_vs_plus_vg[smoothing_kernel].append(variances[smoothing_kernel]['stimuli'][0, 1] / np.sum(variances[smoothing_kernel]['stimuli'][0, 0:2]))
    vs_over_vs_plus_vg[smoothing_kernel].append(variances[smoothing_kernel]['interactions'][0, 1] / np.sum(variances[smoothing_kernel]['interactions'][0, 0:2]))
    # vs_over_vs_plus_vg[smoothing_kernel].append(variances[smoothing_kernel]['mains'][0, 1] / np.sum(variances[smoothing_kernel]['mains'][0, 0:2]))

    # vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['original'][0, 0:2])/np.sum(variances[smoothing_kernel]['original']))
    # vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['mean'][0, 0:2])/np.sum(variances[smoothing_kernel]['mean']))
    # vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['task'][0, 0:2])/np.sum(variances[smoothing_kernel]['task']))
    # vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['difficulty'][0, 0:2])/np.sum(variances[smoothing_kernel]['difficulty']))
    # vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['stimuli'][0, 0:2])/np.sum(variances[smoothing_kernel]['stimuli']))
    # vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['interactions'][0, 0:2])/np.sum(variances[smoothing_kernel]['interactions']))

    vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['original'][0, 0:2]))
    vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['mean'][0, 0:2]))
    vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['task'][0, 0:2]))
    vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['difficulty'][0, 0:2]))
    vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['stimuli'][0, 0:2]))
    vs_plus_vg[smoothing_kernel].append(np.sum(variances[smoothing_kernel]['interactions'][0, 0:2]))


output['vs_over_vs_plus_vg'] = vs_over_vs_plus_vg
output['vs_plus_vg'] = vs_plus_vg
output['variances'] = variances
output['plot_vs+vg'] = {}
output['plot_vs/vs+vg'] = {}

features = ['original', 'mean', 'task', 'difficulty', 'stimuli', 'interactions']
smoothing_kernels = list(vs_over_vs_plus_vg.keys())

plt.figure()
for featureI in np.arange(len(features)):
    vals = []
    for smoothing_kernel in smoothing_kernels:
        vals.append(vs_over_vs_plus_vg[smoothing_kernel][featureI])
    plt.plot(np.arange(len(smoothing_kernels)), vals, label=features[featureI])
    output['plot_vs/vs+vg'][features[featureI]] = vals

plt.xticks(np.arange(len(smoothing_kernels)), labels=smoothing_kernels)
plt.ylabel('vs/vs+vg')
plt.xlabel('factors')
plt.title('vs/vs+vg on each factor')
plt.legend(frameon=False)
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1, step=0.2))
plt.savefig(os.path.join(resultsPath, f'vs_over_vs_plus_vg.png'), dpi=500)


plt.figure()
for featureI in np.arange(len(features)):
    vals = []
    for smoothing_kernel in smoothing_kernels:
        vals.append(vs_plus_vg[smoothing_kernel][featureI])
    plt.plot(np.arange(len(smoothing_kernels)), vals, label=features[featureI])
    output['plot_vs+vg'][features[featureI]] = vals

plt.xticks(np.arange(len(smoothing_kernels)), labels=smoothing_kernels)
plt.ylabel('vs+vg')
plt.xlabel('factors')
plt.title('vs+vg on each factor')
plt.legend(frameon=False)
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1, step=0.2))
plt.savefig(os.path.join(resultsPath, f'vs_plus_vg.png'), dpi=500)
plt.show()

with open(PKL_output, 'wb') as pf:
    pickle.dump(output, pf)

plt.close()
