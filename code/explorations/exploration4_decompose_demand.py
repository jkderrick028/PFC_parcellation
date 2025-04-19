import os.path, pickle, subprocess
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
from sklearn.manifold import MDS
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from py_util_dx.py_utils import setProjectPath, sqmat2vec
from Functional_Fusion.dataset import decompose_pattern_into_group_indiv_noise
from flat2data import flat2ndarray
import SUITPy.flatmap as flatmap
from nitools.cifti import surf_from_cifti

"""
decompose variance into group, individual and noise for the demand dataset. 

last modified: 2024.05.25
"""


projectPath, mainResultsPath = setProjectPath()
# base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
# base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''))
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

output = dict()
output['PFC'] = {}
output['whole_cortex'] = {}
PKL_output = os.path.join(resultsPath, 'output_B4.pkl')

# X_individuals, info_individuals, dataset_obj_individuals = ds.get_dataset(base_dir,
#                                                                           dataset='Demand',
#                                                                           atlas='fs32k',
#                                                                           subj=None,
#                                                                           sess='all',
#                                                                           type='CondHalf')

PKL_Demand = os.path.join(projectPath, 'data', 'Demand_Cond_Half.pkl')
with open(PKL_Demand, 'rb') as pf:
    orig_data = pickle.load(pf)
    X_individuals = orig_data['X_individuals']
    info_individuals = orig_data['info_individuals']
    dataset_obj_individuals = orig_data['dataset_obj_individuals']

part_vec = list(info_individuals[dataset_obj_individuals.part_ind])
cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

# whole cortex
data = flat2ndarray(X_individuals, part_vec, cond_vec)

# fill nans with 0
data[np.isnan(data)] = 0

# mean-centre patterns
for i in np.arange(data.shape[0]):
    for j in np.arange(data.shape[1]):
        mean_pattern = data[i, j].mean(axis=0)
        data[i, j] -= mean_pattern

criterion = 'global'
variances = decompose_pattern_into_group_indiv_noise(data, criterion=criterion)
output['whole_cortex'][criterion] = variances

criterion = 'voxel_wise'
variances = decompose_pattern_into_group_indiv_noise(data, criterion=criterion)
output['whole_cortex'][criterion] = variances

criterion = 'condition_wise'
variances = decompose_pattern_into_group_indiv_noise(data, criterion=criterion)
output['whole_cortex'][criterion] = variances

# PFC
# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

# Read data from two gifti files for left and right hemisphere
glasser_left = os.path.join(surface_helpers_dir, 'glasser.L.label.gii')
glasser_right = os.path.join(surface_helpers_dir, 'glasser.R.label.gii')

flat_surf_L = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'fs_LR.32k.R.flat.surf.gii')

# parcels = ['8BM', '8C', 'IFJp', 'p9-46v', 'a9-46v', 'i6-8', 'AVI']    # original core-frontal parcels
# parcels = ['8BM', 'd32', 'a32pr', '9m', '8BL', '8C', '8Ad', '8Av', 'IFJp', 'IFJa', 'IFSp', 'p9-46v', '9p', '9a', 'a9-46v', '9-46d', '46', 'i6-8', 'AVI', 's6-8', '10d', 'p10p', 'a10p', 'p32', 'SFL', '6ma', 'IFSa', 'FOP4', 'FOP5', '45', '47l', '6r', '44', '47s', 'FOP3', 'MI', 'p47r', 'a47r', '47m', '13l', '11l']
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

# roi L, R hemispheres
label_vec, labels = atlas.get_parcel(gii_files)

# only keep vertices that are within selected ROIs
included_vtx_inds_LR = np.where(label_vec > 0)[0]
included_vtx_inds_L = np.where(label_vec == 1)[0]
included_vtx_inds_R = np.where(label_vec == 2)[0]
excluded_vtx_inds_LR = np.where(label_vec == 0)[0]

data = data[:, :, :, included_vtx_inds_LR]

criterion = 'global'
variances = decompose_pattern_into_group_indiv_noise(data, criterion=criterion)
output['PFC'][criterion] = variances

criterion = 'voxel_wise'
variances = decompose_pattern_into_group_indiv_noise(data, criterion=criterion)
output['PFC'][criterion] = variances

criterion = 'condition_wise'
variances = decompose_pattern_into_group_indiv_noise(data, criterion=criterion)
output['PFC'][criterion] = variances

with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)

