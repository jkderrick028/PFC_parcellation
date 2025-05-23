import numpy as np
import pickle, os
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

# defining ROIs
large_ROI = 'PFC'
# large_ROI = 'visual'
# large_ROI = 'somatosensory'
# large_ROI = 'parietal'

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

output = dict()
PKL_output = os.path.join(resultsPath, f'{dataset_name}_{large_ROI}_output.pkl')

PKL_individualized_parcellation = os.path.join(projectPath, 'results', 'START_A3_bayes_parcellation', dataset_name, f'output_{large_ROI}_masked.pkl')
with open(PKL_individualized_parcellation, 'rb') as pf:
    output_indiv = pickle.load(pf)
    U_indiv = output_indiv['U_indiv']
    U_indiv_label = np.argmax(U_indiv, axis=1) + 1
    U_group_label = output_indiv['U']

n_subjects, n_parcels, P = U_indiv.shape

# count the number of parcels for each individual
n_parcels_per_indiv = [len(np.unique(x)) for x in U_indiv_label]

# count the number of vertices per parcel
n_vertices_per_parcel = np.zeros((n_subjects, n_parcels))
for subjI in np.arange(n_subjects):
    vals, counts = np.unique(U_indiv_label[subjI], return_counts=True)
    n_vertices_per_parcel[subjI] = counts


# for each subject, compute V (n_parcels x n_conditions) then assess how similar these V's are (i.e., to check the power of V)
## loading MDTB data
PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_All_ses-s2.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    X_individuals = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])
# fill nans with 0
X_individuals[np.isnan(X_individuals)] = 0

Y = np.transpose(X_individuals, [0, 2, 1])
V = np.matmul(U_indiv, Y)       # n_subjects x n_parcels x n_conditions
V_simmats = [np.corrcoef(v) for v in V]

output['V'] = V
output['n_parcels_per_indiv'] = n_parcels_per_indiv
output['n_vertices_per_parcel'] = n_vertices_per_parcel

for subjI in np.arange(n_subjects):
    JPG_V_corrmat = os.path.join(resultsPath, f'V_corrmat_sub{subjI}.jpg')
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(V_simmats[subjI], cmap='bwr', vmin=-1, vmax=1)
    ax.set_aspect('equal')
    plt.colorbar(im)
    ax.set_title(f'V corrmat subj {subjI}')
    ax.set_xlabel('parcels')
    ax.set_ylabel('parcels')

    plt.savefig(JPG_V_corrmat, dpi=400, format='jpg')
    plt.close(fig)

with open(PKL_output, 'wb') as pf:
    pickle.dump(output, pf)
