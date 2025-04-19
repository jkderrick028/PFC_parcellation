import os.path, pickle, subprocess
import numpy as np
import Functional_Fusion.atlas_map as am
import nibabel as nib
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from Functional_Fusion.dataset import decompose_pattern_into_group_indiv_noise, flat2ndarray
import SUITPy.flatmap as flatmap
from nitools.cifti import surf_from_cifti

"""
This scripts decomposes the data into group, individual and noise variance components. 
"""


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

# base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
# base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
# base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

output = dict()
output['PFC'] = {}
output['whole_cortex'] = {}
PKL_output = os.path.join(resultsPath, f'{dataset_name}_output.pkl')

# X_individuals, info_individuals, dataset_obj_individuals = ds.get_dataset(base_dir,
#                                                                           dataset='MDTB',
#                                                                           atlas='fs32k',
#                                                                           subj=None,
#                                                                           sess='all',
#                                                                           type='CondHalf')

PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_Half.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    X_individuals = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

part_vec = list(info_individuals[dataset_obj_individuals.part_ind])
cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

# whole cortex
data = flat2ndarray(X_individuals, part_vec, cond_vec)

# fill nans with 0
data[np.isnan(data)] = 0

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

# flat_surf_L = os.path.join(surface_helpers_dir, 'tpl-fs32k_hemi-L_flat.surf.gii')
# flat_surf_R = os.path.join(surface_helpers_dir, 'tpl-fs32k_hemi-R_flat.surf.gii')

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

# visualizing the decomposition results (whole cortex)

# flatmap visualization
underlay_L = os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii')
underlay_R = os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii')
border_LR = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.border')

# normalization_method = 'gse'    # gse or gs
normalization_method = 'gs'

if normalization_method == 'gse':
    voxel_wise_PFC = np.divide(output['PFC']['voxel_wise'], np.tile(np.sum(output['PFC']['voxel_wise'], axis=1).reshape(-1, 1), (1, 3)))
    voxel_wise_cortex = np.divide(output['whole_cortex']['voxel_wise'], np.tile(np.sum(output['whole_cortex']['voxel_wise'], axis=1).reshape(-1, 1), (1, 3)))
else:
    voxel_wise_PFC = np.divide(output['PFC']['voxel_wise'][:, 0:2], np.tile(np.sum(output['PFC']['voxel_wise'][:, 0:2], axis=1).reshape(-1, 1), (1, 2)))
    voxel_wise_cortex = np.divide(output['whole_cortex']['voxel_wise'][:, 0:2], np.tile(np.sum(output['whole_cortex']['voxel_wise'][:, 0:2], axis=1).reshape(-1, 1), (1, 2)))

# v_g
[v_g_extended_L, v_g_extended_R] = surf_from_cifti(atlas.data_to_cifti(voxel_wise_cortex[:, 0].reshape(1, -1)))

figI_flatmap = 11
nHors = 1
nVers = 2
fig, axs = plt.subplots(nHors, nVers, figsize=(15, 12), num=figI_flatmap)
plt.axes(axs[0])
flatmap.plot(v_g_extended_L.reshape(-1, ), surf=flat_surf_L, underlay=underlay_L, alpha=1, cscale=[-0.1, 0.7], borders=border_LR, frame=None, new_figure=False)
axs[0].set_title('cortex L')
plt.axes(axs[1])
flatmap.plot(v_g_extended_R.reshape(-1, ), surf=flat_surf_R, underlay=underlay_R, alpha=1, cscale=[-0.1, 0.7], borders=border_LR, frame=None, new_figure=False, colorbar=True)
axs[1].set_title('cortex R')

# plt.tight_layout()
plt.suptitle('v_g')

PS_variance = os.path.join(resultsPath, f'{dataset_name}_flatmap_v_g_{normalization_method}.png')
if os.path.exists(PS_variance):
    os.remove(PS_variance)
fig.savefig(PS_variance, format='png', dpi=500)

# v_s
[v_s_extended_L, v_s_extended_R] = surf_from_cifti(atlas.data_to_cifti(voxel_wise_cortex[:, 1].reshape(1, -1)))
plt.clf()

fig, axs = plt.subplots(nHors, nVers, figsize=(15, 12), num=figI_flatmap)
plt.axes(axs[0])
flatmap.plot(v_s_extended_L.reshape(-1, ), surf=flat_surf_L, underlay=underlay_L, alpha=1, cscale=[-0.1, 0.7], borders=border_LR, frame=None, new_figure=False)
axs[0].set_title('cortex L')
plt.axes(axs[1])
flatmap.plot(v_s_extended_R.reshape(-1, ), surf=flat_surf_R, underlay=underlay_R, alpha=1, cscale=[-0.1, 0.7], borders=border_LR, frame=None, new_figure=False, colorbar=True)
axs[1].set_title('cortex R')

# plt.tight_layout()
plt.suptitle('v_s')

PS_variance = os.path.join(resultsPath, f'{dataset_name}_flatmap_v_s_{normalization_method}.png')
if os.path.exists(PS_variance):
    os.remove(PS_variance)
fig.savefig(PS_variance, format='png', dpi=500)

# v_e
if normalization_method == 'gse':
    [v_e_extended_L, v_e_extended_R] = surf_from_cifti(atlas.data_to_cifti(voxel_wise_cortex[:, 2].reshape(1, -1)))
    plt.clf()

    fig, axs = plt.subplots(nHors, nVers, figsize=(15, 12), num=figI_flatmap)
    plt.axes(axs[0])
    flatmap.plot(v_e_extended_L.reshape(-1, ), surf=flat_surf_L, underlay=underlay_L, alpha=1, cscale=[-0.1, 0.7], borders=border_LR, frame=None, new_figure=False)
    axs[0].set_title('cortex L')
    plt.axes(axs[1])
    flatmap.plot(v_e_extended_R.reshape(-1, ), surf=flat_surf_R, underlay=underlay_R, alpha=1, cscale=[-0.1, 0.7], borders=border_LR, frame=None, new_figure=False, colorbar=True)
    axs[1].set_title('cortex R')

    # plt.tight_layout()
    plt.suptitle('v_e')

    PS_variance = os.path.join(resultsPath, f'{dataset_name}_flatmap_v_e_{normalization_method}.png')
    if os.path.exists(PS_variance):
        os.remove(PS_variance)
    fig.savefig(PS_variance, format='png', dpi=500)
plt.show()



with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)
