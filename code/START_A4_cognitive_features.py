import os.path, pickle, subprocess
import numpy as np
import pandas as pd
import Functional_Fusion.atlas_map as am
import nibabel as nib
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
import SUITPy.flatmap as flatmap
from nitools.cifti import surf_from_cifti
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

"""
This scripts characterizes the task activation by cognitive features.

modified: 2025.03.11 
 
"""


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

# loading the cognitive feature table
TXT_cognitive_features = os.path.join(projectPath, 'reference_manual', 'featureTable_functionalAtlas.txt')
cognitive_feature_table = pd.read_csv(TXT_cognitive_features, sep='\t', header=0)

# convert the dataframe into numpy array, 47 tasks x 43 features
feature_names = list(cognitive_feature_table.columns)
remove = ['conditionName', 'taskNumUni', 'condNumUni', 'duration']
feature_names = list(filter(lambda x: x not in remove, feature_names))
cognitive_features = cognitive_feature_table.drop(columns = remove).to_numpy()

# z-normalize the cognitive features
cognitive_features = zscore(cognitive_features, axis=0)

PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_All.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    X_individuals = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])
cond_names_data = list(info_individuals['cond_name'])    # a list of 61 conditions, some duplicates
cond_names_data[cond_names_data.index('SpatialMedDiff')] = 'SpatialMapDiff'

cond_names_table = list(cognitive_feature_table['conditionName'])
cond_names_table = [x.strip() for x in cond_names_table]

# print(sorted(set(cond_names_data)))
# print(sorted(cond_names_table))

# average the activities in the data that have similar conditions
X_individuals[np.isnan(X_individuals)] = 0
n_subjects, _, n_vertices = X_individuals.shape

# # subtract out the mean across all the 61 conditions
# X_individuals -= np.mean(X_individuals, axis=1, keepdims=True)

data = np.zeros((n_subjects, len(cond_names_table), n_vertices))
for condI in np.arange(len(cond_names_table)):
    inds = [i for i in np.arange(len(cond_names_data)) if cond_names_data[i] == cond_names_table[condI]]
    inds = np.array(inds)
    data[:, condI, :] = np.mean(X_individuals[:, inds, :], axis=1)


# average vertices within a parcel
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

# large_ROI = 'PFC'
# parcels = ['OFC', '10pp', '10r', '8C', 's6-8', '25', 'p24', 'p47r', '46', 'a10p', '10d', '9m', '8Av', 'IFJp', '10v', '13l', '45', 'i6-8', '9-46d', 'IFJa', '47s', 'SFL', 'a24', 'IFSp', '47m', '9p', '9a', 'pOFC', '8Ad', '11l', 'IFSa', 'a9-46v', '44', 'a47r', '55b', '47l', 's32', 'p9-46v', '8BM', 'p10p', '8BL', 'p32', 'a32pr', 'd32']    # PNAS paper

# large_ROI = 'visual'
# parcels = ['V1', 'V2', 'V3', 'V4']

large_ROI = 'somatosensory'
parcels = ['4', '3a', '3b', '1', '2']

# large_ROI = 'parietal'
# parcels = ['7AL', '7Am', '7Pm', '7PL', 'MIP', 'VIP', '7PC', 'LIPv', 'AIP', 'LIPd']

dict_parcel_vtx = {p:{} for p in parcels}
n_parcels = len(parcels)
data_parcels = np.zeros((n_subjects, len(cond_names_table), n_parcels*2))
parcel_labels = []
for parcelI in np.arange(len(parcels)):
    gii_files = []

    for hemi in ['L', 'R']:
        if hemi == 'L':
            glasser_label = glasser_left
            flat_shape = flat_surf_L
        else:
            glasser_label = glasser_right
            flat_shape = flat_surf_R

        meta = nib.load(flat_shape).meta
        out_gii_file = os.path.join(resultsPath, f'roi_{hemi}.func.gii')
        roi_data = []

        hemi_roi = f'{hemi}_{parcels[parcelI]}'
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

    dict_parcel_vtx[parcels[parcelI]]['included_vtx_inds_L'] = included_vtx_inds_L
    dict_parcel_vtx[parcels[parcelI]]['included_vtx_inds_R'] = included_vtx_inds_R
    dict_parcel_vtx[parcels[parcelI]]['included_vtx_inds_LR'] = included_vtx_inds_LR

    data_parcels[:, :, 2 * parcelI] = np.mean(data[:, :, included_vtx_inds_L], axis=2)    # L
    data_parcels[:, :, 2 * parcelI + 1] = np.mean(data[:, :, included_vtx_inds_R], axis=2)  # R

    parcel_labels.append(f'L_{parcels[parcelI]}')
    parcel_labels.append(f'R_{parcels[parcelI]}')

# non-negative regression
data_parcels_indv = data_parcels - np.mean(data_parcels, axis=1, keepdims=True)
nnls_fitting_results = []
weights = []
for subjI in np.arange(n_subjects):
    reg_nnls = LinearRegression(positive=True, fit_intercept=False).fit(cognitive_features, data_parcels_indv[subjI])
    nnls_fitting_results.append(reg_nnls)
    weights.append(reg_nnls.coef_)

weights = np.array(weights)     # n_subjects x n_parcels x n_cognitive_features (n_parcels contains both L and R hemispheres)


output = dict()
PKL_output = os.path.join(resultsPath, f'{dataset_name}_{large_ROI}_output.pkl')

# get the top 3 cognitive features for each individual
top_3_inds = np.argsort(weights, axis=-1)[:, :, -3:]
output['top_3_inds_indv'] = np.flip(top_3_inds, axis=-1)

# get the top 3 cognitive features for the group
data_parcels_group = data_parcels.mean(axis=0)
data_parcels_group -= np.mean(data_parcels_group, axis=0, keepdims=True)
reg_nnls = LinearRegression(positive=True, fit_intercept=False).fit(cognitive_features, data_parcels_group)
nnls_fitting_results.append(reg_nnls)
weights = reg_nnls.coef_
top_3_inds = np.argsort(weights, axis=-1)[:, -3:]
output['top_3_inds_group'] = np.flip(top_3_inds, axis=-1)

# weights = np.mean(weights, axis=0)
# top_3_inds = np.argsort(weights, axis=-1)[:, -3:]
# output['top_3_inds_group'] = np.flip(top_3_inds, axis=-1)


output['task_names'] = cond_names_table
output['feature_names'] = feature_names
output['nnls_fitting'] = nnls_fitting_results
output['parcel_labels'] = parcel_labels

with open(PKL_output, 'wb') as pf:
    pickle.dump(output, pf)

