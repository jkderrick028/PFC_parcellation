import os.path, pickle
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
from sklearn.manifold import MDS
import nibabel as nb
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
gifti_left = os.path.join(surface_helpers_dir, 'sub-01.L.inflated.32k_fs_LR.surf.gii')
gifti_right = os.path.join(surface_helpers_dir, 'sub-01.R.inflated.32k_fs_LR.surf.gii')

X_group, info_group, dataset_obj_group = ds.get_dataset(base_dir,
                                            dataset='MDTB',
                                            atlas='fs32k',
                                            subj='group',
                                            sess='all',
                                            type='CondAll')

X_individuals, info_individuals, dataset_obj_individuals = ds.get_dataset(base_dir,
                                            dataset='MDTB',
                                            atlas='fs32k',
                                            subj=None,
                                            sess='all',
                                            type='CondAll')
[n_subj, n_tasks, n_vertices] = X_individuals.shape

# COV_task_group = np.matmul((X_group[0] - np.mean(X_group[0], axis=0)), (X_group[0] - np.mean(X_group[0], axis=0)).T)
RDM_task_group = squareform(pdist(X_group[0], 'correlation'))
output['RDM_task_group'] = RDM_task_group

# plotting RDM for tasks (group data), dataset 1 and dataset 2 will both go into this RDM
PS_groupMDS = os.path.join(resultsPath, 'MDS_group.ps')
if os.path.exists(PS_groupMDS):
    os.remove(PS_groupMDS)

figI = 10
fig = plt.figure(figI)

plt.imshow(RDM_task_group, cmap='seismic', vmin=0, vmax=2)
plt.colorbar()
plt.xlabel('tasks')
plt.ylabel('tasks')
plt.title('RDM correlation dist task group')
fig.savefig(PS_groupMDS, format='eps')

# mds for tasks (group data) using correlation distances, dataset 1 and dataset 2 will be separated
PS_mds = os.path.join(resultsPath, 'task_mds_separate.ps')
if os.path.exists(PS_mds):
   os.remove(PS_mds)

nVers = 1
nHors = 2
vmin = 0
vmax = 2

# for dataset 1:
RDM_task_group_1 = squareform(pdist(X_group[0][0:29, :], 'correlation'))
task_labels_1 = info_group.names[0:29]

# for dataset 2:
RDM_task_group_2 = squareform(pdist(X_group[0][29:61, :], 'correlation'))
task_labels_2 = info_group.names[29:61]

fig, ax = plt.subplots(figsize=(11, 12), nrows=nHors, ncols=nVers)

datasets = [1, 2]
coords_2d_combine = []
for datasetI in datasets:
    RDM_this_dataset = eval(f'RDM_task_group_{datasetI}')
    task_labels = list(eval(f'task_labels_{datasetI}'))
    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=0, normalized_stress='auto')
    task_group_2d_coords = embedding.fit_transform(RDM_this_dataset)
    where_rest = [x for x in np.arange(len(task_labels)) if task_labels[x]=='rest']
    coords_rest = task_group_2d_coords[where_rest]
    task_group_2d_coords = task_group_2d_coords - coords_rest
    coords_2d_combine.append(task_group_2d_coords)

    axi = ax.ravel()[datasetI-1]
    axi.scatter(task_group_2d_coords[:, 0], task_group_2d_coords[:, 1])
    axi.set_aspect('equal', 'box')
    axi.set_title(f'dataset {datasetI}')

    # Label points
    indx = 0
    for (i, j) in zip(task_group_2d_coords[:, 0], task_group_2d_coords[:, 1]):
        axi.text(i, j, f'{task_labels[indx]}')
        indx = indx + 1

fig.savefig(PS_mds, format='eps')

output['coords_2d_combine'] = coords_2d_combine
output['task_labels_combine'] = list(info_group.names)

# mds for tasks (group data) using correlation distances, in the mds solutions, overlap rest in dateset 1 and 2
PS_mds = os.path.join(resultsPath, 'task_mds_combined.ps')
if os.path.exists(PS_mds):
   os.remove(PS_mds)

nVers = 1
nHors = 1
vmin = 0
vmax = 2
fig, ax = plt.subplots(figsize=(11, 12), nrows=nHors, ncols=nVers)

coords_2d_combine = np.concatenate(coords_2d_combine)
task_labels = list(info_group.names)

axi = ax
axi.scatter(coords_2d_combine[:, 0], coords_2d_combine[:, 1])
axi.set_aspect('equal', 'box')
axi.set_title(f'dataset 1 and 2')

# Label points
indx = 0
for (i, j) in zip(coords_2d_combine[:, 0], coords_2d_combine[:, 1]):
    axi.text(i, j, f'{task_labels[indx]}')
    indx = indx + 1

fig.savefig(PS_mds, format='eps')

# group RDM variance explained by individual subject's RDMs
RDM_task_individuals = np.zeros((n_subj, n_tasks, n_tasks))
explVar_each_individual = np.zeros((n_subj, 1))
nan_where = np.where(np.isnan(X_individuals))
X_individuals[nan_where] = 0
for subjI in np.arange(n_subj):
    # COV_this_subj = np.matmul((X_individuals[subjI] - np.mean(X_individuals[subjI], axis=0)), (X_individuals[subjI] - np.mean(X_individuals[subjI], axis=0)).T)
    RDM_this_subj = squareform(pdist(X_individuals[subjI], 'correlation'))
    RDM_task_individuals[subjI] = RDM_this_subj
    explVar_each_individual[subjI] = np.corrcoef(sqmat2vec(RDM_this_subj, upperORlower='upper'), sqmat2vec(RDM_task_group, upperORlower='upper'))[0, 1]**2

explVar_mean_acrossSubj = explVar_each_individual.mean()

output['RDM_task_individuals'] = RDM_task_individuals
output['RDM_task_group_1'] = RDM_task_group_1
output['task_labels_1'] = task_labels_1
output['RDM_task_group_2'] = RDM_task_group_2
output['task_labels_2'] = task_labels_2
output['explVar_mean_acrossSubj'] = explVar_mean_acrossSubj

with open(PKL_output, 'wb') as pf:
    pickle.dump(output, pf)

plt.show()
