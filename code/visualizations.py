import os.path, pickle, scipy
import numpy as np
import Functional_Fusion.atlas_map as am
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from py_util_dx.data_utils import get_roi_pacels, get_roi_vtx_from_fs32k
from scipy.stats import ttest_ind
from nitools.cifti import surf_from_cifti
import nitools as nt
import SUITPy.flatmap as flatmap


projectPath, mainResultsPath = setProjectPath()

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

flat_surf_L = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'fs_LR.32k.R.flat.surf.gii')
border_LR = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.border')

def plot_flatmap_labels(label, hemi):
    lid, cmap, names = nt.read_lut(os.path.join(surface_helpers_dir, 'atl-glasser.lut'))

    if hemi == 'L':
        # left cortex
        keep_inds = np.arange(int(cmap.shape[0]/2))
        cmap = np.vstack([np.ones((1, 3)), cmap[keep_inds, :], np.ones((1, 3))])

        flatmap.plot(label.reshape(-1, ),
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

        flatmap.plot(label.reshape(-1, ),
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

