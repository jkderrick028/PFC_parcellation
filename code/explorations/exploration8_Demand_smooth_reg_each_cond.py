import os.path, pickle, subprocess
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import nibabel as nib
from py_util_dx.py_utils import setProjectPath
from sklearn.linear_model import LinearRegression


projectPath, mainResultsPath = setProjectPath()
base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')
# base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''))
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

# load original unsmoothed matrix
PKL_smoothed = os.path.join(mainResultsPath, 'exploration8_Demand_smooth_work', 'smoothed_orig_mm.pkl')

with open(PKL_smoothed, 'rb') as pf:
    output = pickle.load(pf)
    data_unsmoothed = output['X_individuals']

data_unsmoothed = data_unsmoothed[[0]]
data_unsmoothed = data_unsmoothed[:, [0], :]

n_subjects, n_conds, n_vertices = data_unsmoothed.shape

X_individuals = np.copy(data_unsmoothed)

smoothing_kernels = [10, 8, 6, 4, 2]      # mm, fwhm

criterion = 'global'

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

flat_surf_L = os.path.join(surface_helpers_dir, 'tpl-fs32k_hemi-L_flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'tpl-fs32k_hemi-R_flat.surf.gii')

X_individuals_regressed = np.zeros(X_individuals.shape)

# for each subject and condition, save orig image
for subjI in np.arange(n_subjects):
    for condI in np.arange(n_conds):
        X_subj_cond = X_individuals[subjI, condI, :].reshape(1, -1)
        cifti_subj_cond = atlas.data_to_cifti(X_subj_cond)
        cifti_file_name = os.path.join(resultsPath, 'cifti_sub-%02d_cond_%02d_orig.dscalar.nii' % (subjI, condI))
        nib.save(cifti_subj_cond, cifti_file_name)

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
           
            X = atlas.cifti_to_data(cifti_subj_cond_smoothed)
            X[np.where(np.isnan(X))] = 0
            X = X.reshape(-1, 1)
            y = X_subj_cond.reshape(-1, 1)

            # reg = LinearRegression(fit_intercept=False).fit(X,y)
            reg = LinearRegression(fit_intercept=True).fit(X,y)

            X_individuals_regressed[subjI, condI, :] = (reg.coef_ * X + reg.intercept_).flatten()
            # X_individuals_regressed[subjI, condI, :] = (reg.coef_ * X).flatten()
            X_individuals[subjI, condI, :] = X_individuals[subjI, condI, :] - X_individuals_regressed[subjI, condI, :]

            # save leftovers after regressing out
            cifti_subj_cond = atlas.data_to_cifti(X_individuals[subjI, condI, :].reshape(1, -1))
            cifti_file_name = os.path.join(resultsPath, 'cifti_sub-%02d_cond_%02d_res_%dmm.dscalar.nii' % (subjI, condI, smoothing_kernel))
            nib.save(cifti_subj_cond, cifti_file_name)

    PKL_leftover = os.path.join(resultsPath, f'leftover_{smoothing_kernel}_mm.pkl')
    with open(PKL_leftover, 'wb') as pk:
        pickle.dump(X_individuals, pk)

    PKL_regressed = os.path.join(resultsPath, f'regressed_{smoothing_kernel}_mm.pkl')
    with open(PKL_regressed, 'wb') as pk:
        pickle.dump(X_individuals_regressed, pk)

    # for subjI in np.arange(n_subjects):
    #     for condI in np.arange(n_conds):            
    #         cifti_file_name = os.path.join(resultsPath, 'cifti_sub-%02d_cond_%02d.dscalar.nii' % (subjI, condI))
    #         if os.path.exists(cifti_file_name):
    #             os.remove(cifti_file_name)

    #         cifti_smoothed_file_name = os.path.join(resultsPath, 'cifti_sub-%02d_cond_%02d_smoothed_%dmm.dscalar.nii' % (subjI, condI, smoothing_kernel))
    #         if os.path.exists(cifti_smoothed_file_name):
    #             os.remove(cifti_smoothed_file_name)


# save residuals (finer than 2mm)
PKL_smoothed = os.path.join(resultsPath, f'regressed_residuals_mm.pkl')
with open(PKL_smoothed, 'wb') as pk:
    pickle.dump(X_individuals, pk)

# save residual image
for subjI in np.arange(n_subjects):
    for condI in np.arange(n_conds):
        X_subj_cond = X_individuals[subjI, condI, :].reshape(1, -1)
        cifti_subj_cond = atlas.data_to_cifti(X_subj_cond)
        cifti_file_name = os.path.join(resultsPath, 'cifti_sub-%02d_cond_%02d_residuals.dscalar.nii' % (subjI, condI))
        nib.save(cifti_subj_cond, cifti_file_name)
