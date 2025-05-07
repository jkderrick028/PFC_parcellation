import numpy as np
import nitools as nt
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.util as ut
from py_util_dx.py_utils import setProjectPath
import os, pickle
from nitools.cifti import surf_from_cifti
import SUITPy.flatmap as flatmap
import torch
from py_util_dx.data_utils import get_roi_pacels, get_glasser_labels, get_roi_vtx_from_fs32k


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

# appendix = 'whole_cortex'
appendix = 'PFC_masked'

atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

atlas_fname = [os.path.join(surface_helpers_dir, 'glasser.L.label.gii'), os.path.join(surface_helpers_dir, 'glasser.R.label.gii')]
U = atlas.read_data(atlas_fname)
U_shape_orig = U.shape

## dealing with PFC mask
if appendix == 'PFC_masked':
    included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k('PFC')
    # included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(['V1'])
if appendix == 'whole_cortex':
    included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k('whole_cortex')

## Load colormap and labels
lid,cmap,names = nt.read_lut(os.path.join(surface_helpers_dir, 'atl-glasser.lut'))

# if appendix == 'PFC_masked':
#     parcels = get_roi_pacels('PFC')
#     keep_inds = []
#     for i in np.arange(len(names)):
#         if names[i].split('_')[1] in parcels:
#             keep_inds.append(i)
#     lid = lid[keep_inds]
#     cmap = cmap[keep_inds]
#     names = list(map(names.__getitem__, keep_inds))

flat_surf_L = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'fs_LR.32k.R.flat.surf.gii')
border_LR = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.border')

def plot_probseg(label, cmap, hemi):
    if appendix == 'PFC_masked' or appendix == 'whole_cortex':
        # label[included_vtx_inds_LR] -= 1
        label[excluded_vtx_inds_LR] = 181

        # label[excluded_vtx_inds_LR] = 181

    [label_L, label_R] = surf_from_cifti(atlas.data_to_cifti(label.reshape(1, -1)))

    if hemi == 'L':
        # left cortex
        keep_inds = np.arange(int(cmap.shape[0]/2))
        cmap = np.vstack([np.ones((1, 3)), cmap[keep_inds, :], np.ones((1, 3))])

        flatmap.plot(label_L.reshape(-1, ),
                     surf=flat_surf_L,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     label_names=list(map(names.__getitem__, keep_inds)),
                     new_figure=False,
                     frame=None,
                     render='matplotlib',
                     cmap=cmap,
                     borders=border_LR,
                     overlay_type='label',
                     bordersize=3,
                     undermap='gray',
                     underscale=[-1, 0.5]
        )

    else:
        # right cortex
        keep_inds = np.arange(int(cmap.shape[0]/2), cmap.shape[0])
        cmap = np.vstack([np.ones((1, 3)), cmap[keep_inds, :], np.ones((1, 3))])

        flatmap.plot(label_R.reshape(-1, ),
                     surf=flat_surf_R,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     label_names=list(map(names.__getitem__, keep_inds)),
                     new_figure=False,
                     frame=None,
                     render='matplotlib',
                     cmap=cmap,
                     borders=border_LR,
                     overlay_type='label',
                     bordersize=3,
                     undermap='gray',
                     underscale=[-1, 0.5]
        )


# plot the group probabilistic atlas
plt.figure()
plt.subplot(1, 2, 1)
plot_probseg(U.copy(), cmap.copy(), 'L')
plt.subplot(1, 2, 2)
plot_probseg(U.copy(), cmap.copy(), 'R')
plt.suptitle('group')

pass