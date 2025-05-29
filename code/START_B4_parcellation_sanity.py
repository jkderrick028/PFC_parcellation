import numpy as np
import matplotlib.pyplot as plt
import nitools as nt
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
from py_util_dx.data_utils import get_roi_pacels, get_roi_vtx_from_fs32k


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

# defining ROIs
large_ROI = 'PFC'
# large_ROI = 'visual'
# large_ROI = 'somatosensory'
# large_ROI = 'parietal'

## plot DCBC as a function of strengths
strengths = [0.01, 0.1, 0.5, 1.0, 7.0]

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}')
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

dcbc_across_strengths = []
for strength in strengths:
    PKL_dcbc_result = os.path.join(mainResultsPath, 'START_A4_parcellation_evaluation', f'{dataset_name}_{strength}', f'{dataset_name}_{large_ROI}_output.pkl')
    with open(PKL_dcbc_result, 'rb') as pf:
        output_dcbc = pickle.load(pf)
    if len(dcbc_across_strengths) == 0:
        dcbc_across_strengths.append(output_dcbc['output_dcbc_group']['dcbc'])

    dcbc_across_strengths.append(output_dcbc['output_dcbc_indiv']['dcbc'])

dcbc_across_strengths = np.array(dcbc_across_strengths)     # n_strengths x n_subjects
dcbc_across_strengths_mean = np.mean(dcbc_across_strengths, axis=1)
dcbc_across_strengths_ste = np.std(dcbc_across_strengths, axis=1) / np.sqrt(dcbc_across_strengths.shape[1])
strengths = np.concatenate([[0.0], strengths])

fig, ax = plt.subplots(1, 1)
ax.errorbar(np.arange(len(strengths)), dcbc_across_strengths_mean, yerr=dcbc_across_strengths_ste)
ax.set_xticks(ticks=np.arange(len(strengths)), labels=strengths)
ax.set_xlabel('strengths')
ax.set_ylabel('DCBC')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title(f'{large_ROI} data only')
JPG_fig = os.path.join(resultsPath, f'dcbc_across_strengths_{large_ROI}.jpg')
plt.savefig(JPG_fig, dpi=500, format='jpg')



## plotting the spread / concentration of an individualized parcel
strength = 7.0

PKL_individualized_parcellation = os.path.join(projectPath, 'results', 'START_A3_bayes_parcellation', f'{dataset_name}_{strength}', f'output_{large_ROI}_masked.pkl')
with open(PKL_individualized_parcellation, 'rb') as pf:
    output_indiv = pickle.load(pf)
    U_indiv = output_indiv['Uhat_data']
    U_indiv_label = np.argmax(U_indiv, axis=1) + 1
    U_group_label = output_indiv['U']
    V = output_indiv['V'].T             # n_parcels x n_conditions

n_subjects, n_parcels, P = U_indiv.shape

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')
flat_surf_L = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'fs_LR.32k.R.flat.surf.gii')
border_LR = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.border')

atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

def plot_probseg(surf_data, hemi):
    [label_L, label_R] = surf_from_cifti(atlas.data_to_cifti(surf_data.reshape(1, -1)))

    if hemi == 'L':
        # left cortex
        flatmap.plot(label_L.reshape(-1, ),
                     surf=flat_surf_L,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     new_figure=False,
                     frame=None,
                     cmap='bwr',
                     borders=border_LR,
                     bordersize=1,
                     cscale=[-1, 1]
        )

    else:
        # right cortex
        flatmap.plot(label_R.reshape(-1, ),
                     surf=flat_surf_R,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     new_figure=False,
                     frame=None,
                     cmap='bwr',
                     borders=border_LR,
                     bordersize=1,
                     cscale=[-1, 1]
        )


included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(large_ROI)

subjIs = [0]
parcelIs = [3, 4, 5]

for subjI in np.arange(len(subjIs)):
    plt.figure()
    for parI in np.arange(len(parcelIs)):
        surf_data = U_indiv[subjI, parcelIs[parI]]

        plt.subplot(len(parcelIs), 2, 1 + 2 * parI)
        plot_probseg(surf_data, 'L')
        plt.title(f'parcel {parcelIs[parI]} L')

        plt.subplot(len(parcelIs), 2, 2 + 2 * parI)
        plot_probseg(surf_data, 'R')
        plt.title(f'parcel {parcelIs[parI]} R')

    plt.suptitle(f'subject {subjIs[subjI]}')
    JPG_fig = os.path.join(resultsPath, f'parcel_spread_subj_{subjIs[subjI]}.jpg')
    plt.savefig(JPG_fig, dpi=500, format='jpg')



## check the power of V
strength = 1.0

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}_{strength}')
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

output = dict()
PKL_output = os.path.join(resultsPath, f'{dataset_name}_{large_ROI}_output.pkl')

# count the number of parcels for each individual
n_parcels_per_indiv = [len(np.unique(x)) for x in U_indiv_label]

# count the number of vertices per parcel
n_vertices_per_parcel = np.zeros((n_subjects, n_parcels))
for subjI in np.arange(n_subjects):
    vals, counts = np.unique(U_indiv_label[subjI], return_counts=True)
    n_vertices_per_parcel[subjI] = counts

# load V compute similarity between each pair of parcels
V_simmats = np.corrcoef(V)

output['V'] = V
output['V_simmats'] = V_simmats
output['n_parcels_per_indiv'] = n_parcels_per_indiv
output['n_vertices_per_parcel'] = n_vertices_per_parcel

JPG_V_corrmat = os.path.join(resultsPath, f'V_corrmat.jpg')
fig, ax = plt.subplots(1, 1)
im = ax.imshow(V_simmats, cmap='bwr', vmin=-1, vmax=1)
ax.set_aspect('equal')
plt.colorbar(im)
ax.set_title(f'V corrmat')
ax.set_xlabel('parcels')
ax.set_ylabel('parcels')

plt.savefig(JPG_V_corrmat, dpi=400, format='jpg')
plt.close(fig)

with open(PKL_output, 'wb') as pf:
    pickle.dump(output, pf)
