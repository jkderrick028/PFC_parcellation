import os.path, pickle, subprocess
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import nibabel as nib
from py_util_dx.py_utils import setProjectPath


projectPath, mainResultsPath = setProjectPath()
base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''))
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

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

n_conds = len(task_conds)
n_subjects = X_individuals.shape[0]
n_vertices = X_individuals.shape[-1]

smoothing_kernels = [10, 8, 6, 4, 2]      # mm, fwhm

X_individuals_smoothed = np.zeros(X_individuals.shape)

for smoothing_kernel in smoothing_kernels:
    for subjI in np.arange(n_subjects):
        for condI in np.arange(n_conds):
            X_subj_cond = X_individuals[subjI, condI, :].reshape(1, -1)
            cifti_subj_cond = atlas.data_to_cifti(X_subj_cond)
            cifti_file_name = os.path.join(resultsPath, 'cifti_sub-%02d_cond_%02d.dscalar.nii' % (subjI, condI))
            nib.save(cifti_subj_cond, cifti_file_name)

            cifti_smoothed_file_name = os.path.join(resultsPath, 'cifti_sub-%02d_cond_%02d_smoothed_%dmm.dscalar.nii' % (subjI, condI, smoothing_kernel))
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

            cifti_file_name = os.path.join(resultsPath, 'cifti_sub-%02d_cond_%02d.dscalar.nii' % (subjI, condI))
            if os.path.exists(cifti_file_name):
                os.remove(cifti_file_name)

            cifti_smoothed_file_name = os.path.join(resultsPath, 'cifti_sub-%02d_cond_%02d_smoothed_%dmm.dscalar.nii' % (subjI, condI, smoothing_kernel))
            if os.path.exists(cifti_smoothed_file_name):
                os.remove(cifti_smoothed_file_name)
