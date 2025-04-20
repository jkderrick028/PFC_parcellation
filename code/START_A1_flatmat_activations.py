import os.path, subprocess
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import nibabel as nib
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
import SUITPy.flatmap as flatmap
from nitools.cifti import surf_from_cifti


"""

This script makes a flat map of the beta weights for the group data and 3 individuals. 

"""

projectPath, mainResultsPath = setProjectPath()
# base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
# base_dir = '/Users/jkderrick028/jxiang27_graham/scratch/7T_exploration/data/FunctionalFusion'
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''))
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

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

parcels = ['8BM', 'd32', 'a32pr', '9m', '8BL', '8C', '8Ad', '8Av', 'IFJp', 'IFJa', 'IFSp', 'p9-46v', '9p', '9a', 'a9-46v', '9-46d', '46', 'i6-8', 'AVI', 's6-8', '10d', 'p10p', 'a10p', 'p32', 'SFL', '6ma', 'IFSa', 'FOP4', 'FOP5', '45', '47l', '6r', '44', '47s', 'FOP3', 'MI', 'p47r', 'a47r', '47m', '13l', '11l']
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


# loading group data condAll
X_group, info_group, dataset_obj_group = ds.get_dataset(base_dir,
                                                        dataset='MDTB',
                                                        atlas='fs32k',
                                                        subj='group',
                                                        sess='all',
                                                        type='CondAll')

# loading individual data condAll
X_individuals, info_individuals, dataset_obj_individuals = ds.get_dataset(base_dir,
                                                                          dataset='MDTB',
                                                                          atlas='fs32k',
                                                                          subj=None,
                                                                          sess='all',
                                                                          type='CondAll')

n_task_conditions = X_individuals.shape[1]

# flatmap visualization
underlay_L = os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii')
underlay_R = os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii')
border_LR = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.border')

# roi visualization, L, R hemispheres
label_vec, labels = atlas.get_parcel(gii_files)
[roi_extended_L, roi_extended_R] = surf_from_cifti(atlas.data_to_cifti(label_vec.reshape(1, -1)))

figI_flatmap = 11
nHors = 1
nVers = 2
fig, axs = plt.subplots(nHors, nVers, figsize=(15, 12), num=figI_flatmap)
plt.axes(axs[0])
flatmap.plot(roi_extended_L.reshape(-1, ), surf=flat_surf_L, underlay=underlay_L, alpha=1, borders=border_LR, frame=None, new_figure=False)
axs[0].set_title('ROI L')
plt.axes(axs[1])
flatmap.plot(roi_extended_R.reshape(-1, ), surf=flat_surf_R, underlay=underlay_R, alpha=1, borders=border_LR, frame=None, new_figure=False)
axs[1].set_title('ROI R')

PS_roi = os.path.join(resultsPath, 'flatmap_roi.png')
if os.path.exists(PS_roi):
    os.remove(PS_roi)

fig.tight_layout()
# fig.savefig(PS_roi, format='eps')
fig.savefig(PS_roi, format='png')

# task activations, L hemisphere
nHors = 4   # 1 group + 3 example subjects
nVers = 4   # 4 example task conditions per page
nPages = np.ceil(n_task_conditions/nVers).astype(int)

example_subjIDs = [0, 1, 2]
task_names = list(info_group.names)
for hemi in ['L', 'R']:
    for pageI in np.arange(nPages):
        start_cond_ind = pageI * nVers
        end_cond_ind = np.min([(pageI+1)*nVers, n_task_conditions])

        plt.clf()
        fig, axs = plt.subplots(nHors, nVers, num=figI_flatmap, figsize=(15, 16))
        fig.suptitle(f'cond {start_cond_ind} to {end_cond_ind}, {hemi}')
        colI = 0
        # group data
        for condI in np.arange(start_cond_ind, end_cond_ind):
            [X_group_extended_L, X_group_extended_R] = surf_from_cifti(atlas.data_to_cifti(X_group[0, condI, :].reshape(1, -1)))
            if hemi == 'L':
                X_group_extended = X_group_extended_L
                flat_surf = flat_surf_L
                underlay = underlay_L
                frame = [-165, 25, -90, 140]
            else:
                X_group_extended = X_group_extended_R
                flat_surf = flat_surf_R
                underlay = underlay_R
                frame = [-20, 170, -110, 120]

            plt.axes(axs[0, colI])
            flatmap.plot(X_group_extended.reshape(-1, ), surf=flat_surf, underlay=underlay, alpha=1, borders=None,
                         frame=frame, cscale=[-0.1, 0.1], new_figure=False)
            axs[0, colI].set_title('group cond%02d %s' % (condI, task_names[condI]))

            for subjI in np.arange(len(example_subjIDs)):
                [X_indv_extended_L, X_indv_extended_R] = surf_from_cifti(atlas.data_to_cifti(X_individuals[example_subjIDs[subjI], condI, :].reshape(1, -1)))
                if hemi == 'L':
                    X_indv_extended = X_indv_extended_L
                    frame = [-165, 25, -90, 140]
                else:
                    X_indv_extended = X_indv_extended_R
                    frame = [-20, 170, -110, 120]

                plt.axes(axs[1+subjI, colI])
                flatmap.plot(X_indv_extended.reshape(-1, ), surf=flat_surf, underlay=underlay, alpha=1, borders=None,
                             frame=frame, cscale=[-0.2, 0.2], new_figure=False)
                axs[1+subjI, colI].set_title('subj%02d cond%02d %s' % (example_subjIDs[subjI], condI, task_names[condI]))
            colI = colI + 1

        plt.tight_layout()
        PS_task_flatmap = os.path.join(resultsPath, f'flatmap_task_page_{pageI}_{hemi}.png')
        if os.path.exists(PS_task_flatmap):
            os.remove(PS_task_flatmap)

        fig.savefig(PS_task_flatmap, format='png')

plt.show()
