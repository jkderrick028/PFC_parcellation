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

# parcels = ['8BM', '8C', 'IFJp', 'p9-46v', 'a9-46v', 'i6-8', 'AVI']    # original core-frontal parcels
parcels = ['OFC', '10pp', '10r', '8C', 's6-8', '25', 'p24', 'p47r', '46', 'a10p', '10d', '9m', '8Av', 'IFJp', '10v', '13l', '45', 'i6-8', '9-46d', 'IFJa', '47s', 'SFL', 'a24', 'IFSp', '47m', '9p', '9a', 'pOFC', '8Ad', '11l', 'IFSa', 'a9-46v', '44', 'a47r', '55b', '47l', 's32', 'p9-46v', '8BM', 'p10p', '8BL', 'p32', 'a32pr', 'd32']    # PNAS paper
gii_files = []

for hemi in ['L', 'R']:
    if hemi == 'L':
        glasser_label = glasser_left
        flat_shape = os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii')
    else:
        glasser_label = glasser_right
        flat_shape = os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii')

    meta = nib.load(flat_shape).meta
    out_gii_file = os.path.join(resultsPath, f'roi_{hemi}.func.gii')
    roi_data = []
    for roi in parcels:
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

# X_individuals, info_individuals, dataset_obj_individuals = ds.get_dataset(base_dir,
#                                                                           dataset='MDTB',
#                                                                           atlas='fs32k',
#                                                                           subj=None,
#                                                                           sess='all',
#                                                                           type='CondHalf')

smoothing_resultsPath = os.path.join(mainResultsPath, 'exploration6_smooth_work')
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

n_conds = len(task_conds)
n_subjects = X_individuals.shape[0]
n_vertices = X_individuals.shape[-1]

smoothing_kernels = [10, 8, 6, 4, 2, 'residuals']      # mm, fwhm

for smoothing_kernel in smoothing_kernels:
    output[f'{smoothing_kernel}_mm'] = {}

criterion = 'global'

for smoothing_kernel in smoothing_kernels:
    PKL_smoothed = os.path.join(smoothing_resultsPath, f'smoothed_{smoothing_kernel}_mm.pkl')
    with open(PKL_smoothed, 'rb') as pf:
        X_individuals_smoothed = pickle.load(pf)

    # whole cortex
    data = flat2ndarray(X_individuals_smoothed, part_vec, cond_vec)

    variances = decompose_pattern_into_group_indiv_noise(data, criterion=criterion)
    output[f'{smoothing_kernel}_mm']['whole_cortex'] = variances

    # PFC
    data = data[:, :, :, included_vtx_inds_LR]

    variances = decompose_pattern_into_group_indiv_noise(data, criterion=criterion)
    output[f'{smoothing_kernel}_mm']['PFC'] = variances

    X_individuals = X_individuals - X_individuals_smoothed

    print(f'finished decomposing smoothed_{smoothing_kernel}_mm')


with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)
