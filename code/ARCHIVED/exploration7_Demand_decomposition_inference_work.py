import os.path, pickle, subprocess
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from py_util_dx.py_utils import setProjectPath, sqmat2vec
from sklearn.linear_model import LinearRegression
from Functional_Fusion.dataset import decompose_pattern_into_group_indiv_noise
from flat2data import flat2ndarray
import SUITPy.flatmap as flatmap
from nitools.cifti import surf_from_cifti


projectPath, mainResultsPath = setProjectPath()
# base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
# base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''))
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

output = dict()
PKL_output = os.path.join(resultsPath, 'output.pkl')

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

# Read data from two gifti files for left and right hemisphere
glasser_left = os.path.join(surface_helpers_dir, 'glasser.L.label.gii')
glasser_right = os.path.join(surface_helpers_dir, 'glasser.R.label.gii')

flat_surf_L = os.path.join(surface_helpers_dir, 'tpl-fs32k_hemi-L_flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'tpl-fs32k_hemi-R_flat.surf.gii')

X_individuals, info_individuals, dataset_obj_individuals = ds.get_dataset(base_dir,
                                                                          dataset='Demand',
                                                                          atlas='fs32k',
                                                                          subj=None,
                                                                          sess='all',
                                                                          type='CondAll')

X_individuals[np.where(np.isnan(X_individuals))] = 0
task_conds = list(info_individuals.names)

task_conds = [k.split('_') for k in task_conds]
conds_task_demand = [k[0] for k in task_conds]
conds_difficulty = [k[1] for k in task_conds]
conds_stimuli = [k[2] for k in task_conds]

n_subjects = X_individuals.shape[0]
n_vertices = X_individuals.shape[-1]

smoothing_kernels = [10, 8, 6, 4, 2]      # mm, fwhm

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
output['orig']['whole_cortex'] = {}

for smoothing_kernel in smoothing_kernels:
    output[f'{smoothing_kernel}_mm'] = {}
    output[f'{smoothing_kernel}_mm']['whole_cortex'] = {}

for region in parcel_dict.keys():
    output['orig'][region] = {}
    for smoothing_kernel in smoothing_kernels:
        output[f'{smoothing_kernel}_mm'][region] = {}

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


criterion = 'global'

np.random.default_rng(seed=0)
n_subjects_per_sample = 10
n_samples = 20
subjectInds_sampled = np.zeros((n_samples, n_subjects_per_sample))
for sampleI in np.arange(n_samples):
    subjectInds_sampled[sampleI, :] = np.random.choice(n_subjects, n_subjects_per_sample, replace=False)
subjectInds_sampled = subjectInds_sampled.astype(int)

smoothing_kernels.insert(0, 'orig')

conds_criteria = ['task_demand', 'difficulty', 'stimuli']

for smoothing_kernel in smoothing_kernels:
    if smoothing_kernel == 'orig':
        X_individuals_smoothed = X_individuals
    else:
        PKL_smoothed = os.path.join(mainResultsPath, 'exploration7_Demand_smooth_work', f'smoothed_{smoothing_kernel}_mm.pkl')
        with open(PKL_smoothed, 'rb') as pf:
            X_individuals_smoothed = pickle.load(pf)

    for conds_criterion in conds_criteria:
        cond_vec = eval(f'conds_{conds_criterion}')
        if conds_criterion == 'task_demand':
            cond_vec = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
            part_vec = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        elif conds_criterion == 'difficulty':
            cond_vec = [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]
            part_vec = [1, 2, 1, 2, 3, 4, 3, 4, 5, 6, 5, 6]
        elif conds_criterion == 'stimuli':
            cond_vec = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
            part_vec = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
        else:
            cond_vec = []
            part_vec = []
            print('error')

        # whole cortex
        data = flat2ndarray(X_individuals_smoothed, part_vec, cond_vec)

        variances = np.zeros((n_samples+1, 3))
        variances[0, :] = decompose_pattern_into_group_indiv_noise(data, criterion=criterion)
        for sampleI in np.arange(n_samples):
            variances[sampleI+1, :] = decompose_pattern_into_group_indiv_noise(data[subjectInds_sampled[sampleI, :]], criterion=criterion)

        if smoothing_kernel == 'orig':
            output[smoothing_kernel]['whole_cortex'][conds_criterion] = variances
        else:
            output[f'{smoothing_kernel}_mm']['whole_cortex'][conds_criterion] = variances

        # regions
        for region in parcel_dict.keys():
            data_region = data[:, :, :, included_vtx_inds_LR_dict[region]]

            variances = np.zeros((n_samples + 1, 3))
            variances[0, :] = decompose_pattern_into_group_indiv_noise(data_region, criterion=criterion)
            for sampleI in np.arange(n_samples):
                variances[sampleI + 1, :] = decompose_pattern_into_group_indiv_noise(data_region[subjectInds_sampled[sampleI, :]], criterion=criterion)
            if smoothing_kernel == 'orig':
                output[smoothing_kernel][region][conds_criterion] = variances
            else:
                output[f'{smoothing_kernel}_mm'][region][conds_criterion] = variances

    X_individuals = X_individuals - X_individuals_smoothed

    if smoothing_kernel == 'orig':
        print('finished decomposing original data')
    else:
        print(f'finished decomposing smoothed_{smoothing_kernel}_mm')

with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)
