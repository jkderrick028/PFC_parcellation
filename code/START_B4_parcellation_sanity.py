import numpy as np
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
from py_util_dx.py_utils import setProjectPath
import os, pickle
from visualizations import flatmap_real_vals, plot_flatmap_labels
from nitools.cifti import surf_from_cifti
from py_util_dx.data_utils import convert_prob_atlas_to_absolute_labels, get_roi_vtx_from_fs32k
from scipy.spatial.distance import pdist


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

# defining ROIs
# large_ROI = 'PFC'
large_ROI = 'visual'
# large_ROI = 'somatosensory'
# large_ROI = 'parietal'

atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

## plot DCBC as a function of strengths
# strengths = [0.01, 0.1, 0.5, 1.0, 7.0]
strengths = [7.0]

included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(large_ROI)

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
    labels_in_glasser = output_indiv['labels_in_glasser']
    parcel_names_in_glasser = output_indiv['parcel_names_in_glasser']

    U_indiv = output_indiv['Uhat_data']
    U_indiv_label = convert_prob_atlas_to_absolute_labels(U_indiv, labels_in_glasser, excluded_vtx_inds_LR)
    U_group = output_indiv['U_roi']
    U_group_label = convert_prob_atlas_to_absolute_labels(U_group, labels_in_glasser, excluded_vtx_inds_LR).flatten()

    V = output_indiv['V'].T             # n_parcels x n_conditions

n_subjects, n_parcels, P = U_indiv.shape

## for mean across subjects
U_indiv_mean_across_subjects = np.mean(U_indiv, axis=0)

# computing the correlation of the individualized atlas and the glasser group atlas
cosine_indiv_glasser = np.zeros((n_parcels,))
for parI in np.arange(n_parcels):
    indiv_parcellation = U_indiv_mean_across_subjects[parI, included_vtx_inds_LR]
    group_parcellation = U_group[parI, included_vtx_inds_LR]
    # corr_indiv_glasser[parI] = np.corrcoef(indiv_parcellation, group_parcellation)[0, 1]
    cosine_indiv_glasser[parI] = 1 - pdist(np.array([indiv_parcellation, group_parcellation]), metric='cosine')


plt.figure()

for parI in np.arange(n_parcels):
    group_label = 181 * np.ones(U_group_label.shape)
    group_label = group_label.astype(int)
    group_label[U_group_label == labels_in_glasser[parI]] = labels_in_glasser[parI]
    [label_L, label_R] = surf_from_cifti(atlas.data_to_cifti(group_label.reshape(1, -1)))

    plt.clf()
    plt.subplot(2, 2, 1)
    plot_flatmap_labels(label_L, 'L')

    plt.subplot(2, 2, 2)
    plot_flatmap_labels(label_R, 'R')

    surf_data = U_indiv_mean_across_subjects[parI]
    plt.subplot(2, 2, 3)
    flatmap_real_vals(surf_data, 'L')

    plt.subplot(2, 2, 4)
    flatmap_real_vals(surf_data, 'R')

    plt.tight_layout()

    plt.suptitle(f'mean across subjects, parcel {parcel_names_in_glasser[parI]}')
    JPG_fig = os.path.join(resultsPath, f'parcel_spread_subj_mean_{parcel_names_in_glasser[parI]}.jpg')
    plt.savefig(JPG_fig, dpi=500, format='jpg')


## check the power of V
strength = 20.0

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
    n_vertices_per_parcel[subjI] = counts[1:]

# load V compute similarity between each pair of parcels
V_simmats = np.corrcoef(V)

output['V'] = V
output['V_simmats'] = V_simmats
output['n_parcels_per_indiv'] = n_parcels_per_indiv
output['n_vertices_per_parcel'] = n_vertices_per_parcel
output['cosine_indiv_glasser'] = cosine_indiv_glasser

JPG_V_corrmat = os.path.join(resultsPath, f'V_corrmat_{large_ROI}.jpg')
fig, ax = plt.subplots(1, 1)
im = ax.imshow(V_simmats, cmap='bwr', vmin=-1, vmax=1)
ax.set_aspect('equal')
plt.colorbar(im)
ax.set_title(f'V corrmat {large_ROI}')
ax.set_xlabel('parcels')
ax.set_ylabel('parcels')

plt.savefig(JPG_V_corrmat, dpi=400, format='jpg')
plt.close(fig)

with open(PKL_output, 'wb') as pf:
    pickle.dump(output, pf)
