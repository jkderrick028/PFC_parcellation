import os.path, pickle, subprocess
import numpy as np
import Functional_Fusion.atlas_map as am
import nibabel as nib
from py_util_dx.py_utils import setProjectPath
from Functional_Fusion.dataset import decompose_pattern_into_group_indiv_noise, flat2ndarray
from py_util_dx.data_utils import get_roi_vtx_from_fs32k

"""
This script splits the data into frequency bands and decomposes the variance of each frequency band into group, subject, noise
"""


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

output = dict()
PKL_output = os.path.join(resultsPath, 'output.pkl')

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

flat_surf_L = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'fs_LR.32k.R.flat.surf.gii')

PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_Half.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    X_individuals = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

part_vec = list(info_individuals['half'])
cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

X_individuals[np.where(np.isnan(X_individuals))] = 0

## smoothing
# save orig, no smoothing
PKL_smoothed = os.path.join(resultsPath, f'smoothed_orig_mm.pkl')
with open(PKL_smoothed, 'wb') as pk:
    pickle.dump(X_individuals, pk)

n_subjects, n_conds, n_vertices = X_individuals.shape

smoothing_kernels = [10, 8, 6, 4, 2]      # mm, fwhm

criterion = 'global'

X_individuals_smoothed = np.zeros(X_individuals.shape)

temp_smoothing_path = os.path.join(resultsPath, 'temp')
if not os.path.exists(temp_smoothing_path):
    os.makedirs(temp_smoothing_path)

for smoothing_kernel in smoothing_kernels:
    for subjI in np.arange(n_subjects):
        for condI in np.arange(n_conds):
            X_subj_cond = X_individuals[subjI, condI, :].reshape(1, -1)
            cifti_subj_cond = atlas.data_to_cifti(X_subj_cond)
            cifti_file_name = os.path.join(temp_smoothing_path, 'cifti_sub-%02d_cond_%02d.dscalar.nii' % (subjI, condI))
            nib.save(cifti_subj_cond, cifti_file_name)

            cifti_smoothed_file_name = os.path.join(temp_smoothing_path, 'cifti_sub-%02d_cond_%02d_smoothed_%dmm.dscalar.nii' % (subjI, condI, smoothing_kernel))
            wb_cmd = f'wb_command -cifti-smoothing {cifti_file_name} {smoothing_kernel} {smoothing_kernel} COLUMN {cifti_smoothed_file_name} -left-surface {flat_surf_L} -right-surface {flat_surf_R} -fwhm'
            subprocess.run(wb_cmd, shell=True)

            cifti_subj_cond_smoothed = nib.load(cifti_smoothed_file_name)
            X_individuals_smoothed[subjI, condI, :] = atlas.cifti_to_data(cifti_subj_cond_smoothed)

    X_individuals_smoothed[np.where(np.isnan(X_individuals_smoothed))] = 0

    PKL_smoothed = os.path.join(resultsPath, f'smoothed_{smoothing_kernel}_mm.pkl')
    with open(PKL_smoothed, 'wb') as pk:
        pickle.dump(X_individuals_smoothed, pk)

    for subjI in np.arange(n_subjects):
        for condI in np.arange(n_conds):
            X_individuals[subjI, condI, :] = X_individuals[subjI, condI, :] - X_individuals_smoothed[subjI, condI, :]

            cifti_file_name = os.path.join(temp_smoothing_path, 'cifti_sub-%02d_cond_%02d.dscalar.nii' % (subjI, condI))
            if os.path.exists(cifti_file_name):
                os.remove(cifti_file_name)

            cifti_smoothed_file_name = os.path.join(temp_smoothing_path, 'cifti_sub-%02d_cond_%02d_smoothed_%dmm.dscalar.nii' % (subjI, condI, smoothing_kernel))
            if os.path.exists(cifti_smoothed_file_name):
                os.remove(cifti_smoothed_file_name)

# save residuals (finer than 2mm)
PKL_smoothed = os.path.join(resultsPath, f'smoothed_residuals_mm.pkl')
with open(PKL_smoothed, 'wb') as pk:
    pickle.dump(X_individuals, pk)


## decomposition
smoothing_kernels = ['orig', 10, 8, 6, 4, 2, 'residuals']      # mm, fwhm

ROIs = ['PFC', 'parietal', 'visual', 'somatosensory']
n_rois = len(ROIs)
included_vtx_inds_LR_dict = {}

for roi in ROIs:
    included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(roi)
    included_vtx_inds_LR_dict[roi] = included_vtx_inds_LR

criterion = 'global'

for smoothing_kernel in smoothing_kernels:
    output[smoothing_kernel] = {}

    PKL_smoothed = os.path.join(resultsPath, f'smoothed_{smoothing_kernel}_mm.pkl')
    with open(PKL_smoothed, 'rb') as pf:
        X_individuals_smoothed = pickle.load(pf)

    data = flat2ndarray(X_individuals_smoothed, part_vec, cond_vec)

    # regions
    for roi in ROIs:
        data_region = data[:, :, :, included_vtx_inds_LR_dict[roi]]
        variances = decompose_pattern_into_group_indiv_noise(data_region, criterion=criterion)

        if smoothing_kernel in ['orig', 'residuals']:
            output[smoothing_kernel][roi] = variances
        else:
            output[f'{smoothing_kernel}_mm'][roi] = variances

    if smoothing_kernel in ['orig', 'residuals']:
        print(f'finished decomposing {smoothing_kernel}')
    else:
        print(f'finished decomposing smoothed_{smoothing_kernel}_mm')

with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)

