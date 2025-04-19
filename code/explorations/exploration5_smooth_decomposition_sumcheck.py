import os.path, pickle, subprocess
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
from sklearn.manifold import MDS
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
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
# base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
# base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''))
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

smooth_decomposition_results_path = os.path.join(mainResultsPath, 'exploration5_smooth_decomposition', 'all_subjects')

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

X_individuals, info_individuals, dataset_obj_individuals = ds.get_dataset(base_dir,
                                                                          dataset='MDTB',
                                                                          atlas='fs32k',
                                                                          subj=['sub-02'],
                                                                          sess='all',
                                                                          type='CondHalf')
X_individuals[np.where(np.isnan(X_individuals))] = 0

data_original = X_individuals[:, 0, :]
cifti_original_file = os.path.join(resultsPath, f'cifti_sub-00_cond_00_original.dscalar.nii')
nib.save(atlas.data_to_cifti(data_original), cifti_original_file)

smoothing_kernels = [10, 4]

cifti_file_name_smoothed_10mm = os.path.join(smooth_decomposition_results_path, f'cifti_sub-00_cond_00_smoothed_10mm.dscalar.nii')
data_smoothed_10mm = atlas.cifti_to_data(nib.load(cifti_file_name_smoothed_10mm))

cifti_file_name_smoothed_4mm = os.path.join(smooth_decomposition_results_path, f'cifti_sub-00_cond_00_smoothed_4mm.dscalar.nii')
data_smoothed_4mm = atlas.cifti_to_data(nib.load(cifti_file_name_smoothed_4mm))

cifti_file_name_residuals_4mm = os.path.join(smooth_decomposition_results_path, f'cifti_sub-00_cond_00_4mm_residuals.dscalar.nii')
data_residuals_4mm = atlas.cifti_to_data(nib.load(cifti_file_name_residuals_4mm))

sum_check = X_individuals[:, 0, :] - (data_smoothed_10mm + data_smoothed_4mm + data_residuals_4mm)

plt.plot(sum_check.flatten())
fig_file_name = os.path.join(resultsPath, 'diff.png')
plt.savefig(fig_file_name, format='png', dpi=500)

sum_check[np.isnan(sum_check)] = 0
print(np.max(np.abs(sum_check)))
