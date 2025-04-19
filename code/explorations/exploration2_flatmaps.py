import itertools
import os.path, pickle, subprocess
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
from sklearn.manifold import MDS
import nibabel as nib
import SUITPy.flatmap as flatmap
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from py_util_dx.py_utils import setProjectPath, sqmat2vec


projectPath, mainResultsPath = setProjectPath()
base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
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

parcels = ['8BM', '8C', 'IFJp', 'p9-46v', 'a9-46v', 'i6-8', 'AVI']
gii_files = []

for hemi in ['L', 'R']:
    if hemi == 'L':
        glasser_label = glasser_left
        meta = {'AnatomicalStructurePrimary': 'CortexLeft', 'AnatomicalStructureSecondary': 'Invalid', 'GeometricType': 'Flat'}
    else:
        glasser_label = glasser_right
        meta = {'AnatomicalStructurePrimary': 'CortexRight', 'AnatomicalStructureSecondary': 'Invalid',
                'GeometricType': 'Flat'}

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

X_group, info_group, dataset_obj_group = ds.get_dataset(base_dir,
                                            dataset='MDTB',
                                            atlas='fs32k',
                                            subj='group',
                                            sess='all',
                                            type='CondAll')



print(X_group)

