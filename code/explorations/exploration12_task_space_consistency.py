import os.path, pickle, subprocess, rsatoolbox
import numpy as np
import Functional_Fusion.atlas_map as am
import nibabel as nib
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from Functional_Fusion.dataset import decompose_pattern_into_group_indiv_noise
from flat2data import flat2ndarray

"""
This function examines the consistency of response patterns and RDMs across subjects for given ROIs. RDMs are computed using crossnobis distances. 

What we are interested in is whether PFC regions show high consistency of task RDMs given low consistency of response patterns. 

last modified: 2024.08.31
"""


projectPath, mainResultsPath = setProjectPath()
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

PKL_data = os.path.join(projectPath, 'data', 'FunctionalFusion_Cond_Half.pkl')

with open(PKL_data, 'rb') as pf:
    temp = pickle.load(pf)
    X_individuals = temp['X_individuals']
    info_individuals = temp['info_individuals']
    dataset_obj_individuals = temp['dataset_obj_individuals']

X_individuals[np.where(np.isnan(X_individuals))] = 0
task_conds = list(info_individuals.names)

part_vec = list(info_individuals[dataset_obj_individuals.part_ind])
cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

data = flat2ndarray(X_individuals, part_vec, cond_vec)

# select a subset of conditions
cond_inds = [0, 1, 20, 28]
data = data[:, :, cond_inds, :]
temp_inds = [x for x in np.arange(len(cond_vec)) if (cond_vec[x]-1) in cond_inds]
part_vec = list(map(part_vec.__getitem__, temp_inds))
cond_vec = list(map(cond_vec.__getitem__, temp_inds))
X_individuals = X_individuals[:, temp_inds, :]

n_conds = data.shape[2]
n_subjects = X_individuals.shape[0]
n_vertices = X_individuals.shape[-1]

parcel_dict = {
    'DLPFC': ['9a', '9-46d', '9p', 'SFL', '8BL', 's6-8', '8Ad', 'i6-8', 'a9-46v', '46', '8Av', 'p9-46v', '8C'],
    'somatosensory': ['4', '3a', '3b', '1', '2'],
    'parietal': ['7AL', '7Am', '7Pm', '7PL', 'MIP', 'VIP', '7PC', 'LIPv', 'AIP', 'LIPd'],
    'visual': ['V1', 'V2', 'V3', 'V4']
}

included_vtx_inds_LR_dict = {}

for region in parcel_dict.keys():
    included_vtx_inds_LR_dict[region] = []

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


# variance decomposition 
criterion = 'global'
output['variances'] = {}
output['RDM_consistency'] = {}
n_partitions = data.shape[1]

for region in parcel_dict.keys():
    data_region = data[:, :, :, included_vtx_inds_LR_dict[region]]
    output['variances'][region] = decompose_pattern_into_group_indiv_noise(data_region, criterion=criterion)

# crossnobis distances
nVers = 2
nHors = np.ceil(len(parcel_dict.keys())/nVers).astype(int)
figI = 10
fig, ax = plt.subplots(nrows=nHors, ncols=nVers, figsize=(15, 15))
currSubplotI = 0
fig.suptitle('RDM consistency across subjects')
for region in parcel_dict.keys():
    data_region = X_individuals[:, :, included_vtx_inds_LR_dict[region]]
    rdms_across_subjects = []
    for subjectI in np.arange(n_subjects):
        measurements = data_region[subjectI]
        des = {'subj': subjectI}
        obs_des = {'conds': cond_vec, 'parts': part_vec}

        data_thisSubject = rsatoolbox.data.dataset.Dataset(measurements=measurements,
                                                           descriptors=des,
                                                           obs_descriptors=obs_des
                                                           )
        rdm_cv = rsatoolbox.rdm.calc_rdm(data_thisSubject, method='crossnobis', descriptor='conds', cv_descriptor='parts')
        rdms_across_subjects.append(rdm_cv.get_vectors().squeeze())

    rdms_across_subjects = np.array(rdms_across_subjects)
    corrmat = np.corrcoef(rdms_across_subjects)
    im = ax.ravel()[currSubplotI].imshow(corrmat, cmap='bwr', vmin=-1, vmax=1)
    ax.ravel()[currSubplotI].set_title(region)
    ax.ravel()[currSubplotI].set_xlabel('subjects')
    ax.ravel()[currSubplotI].set_ylabel('subjects')
    plt.colorbar(im, ax=ax.ravel()[currSubplotI])
    currSubplotI += 1

    output['RDM_consistency'][region] = corrmat

plt.tight_layout()

PS_rdms = os.path.join(resultsPath, 'task_rdm_consistency.png')
plt.savefig(PS_rdms, format='png', dpi=500)

with open(PKL_output, 'wb') as pk:
    pickle.dump(output, pk)

plt.close(fig)
