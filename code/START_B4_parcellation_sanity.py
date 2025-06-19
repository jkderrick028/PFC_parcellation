import numpy as np
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
from py_util_dx.py_utils import setProjectPath
import os, pickle
from visualizations import flatmap_real_vals, plot_flatmap_labels
from nitools.cifti import surf_from_cifti
from py_util_dx.data_utils import get_roi_vtx_from_fs32k
from scipy.spatial.distance import pdist


def run_sanity_check(ROI):
    """
    1. plotting the spread of each parcel
    2. computing the power of V

    Args:
        ROI: str
            'PFC', 'visual', 'somatosensory', 'parietal'
    """

    projectPath, mainResultsPath = setProjectPath()

    dataset_name = 'MDTB' # or Demand

    atlas_str = 'fs32k'
    atlas, ainf = am.get_atlas(atlas_str)

    included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(ROI)

    resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}')
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    ## plotting the spread / concentration of an individualized parcel
    PKL_individualized_parcellation = os.path.join(projectPath, 'results', 'START_A3_bayes_parcellation', f'{dataset_name}', f'output_{ROI}.pkl')
    with open(PKL_individualized_parcellation, 'rb') as pf:
        output_indiv = pickle.load(pf)
        labels_in_glasser = output_indiv['labels_in_glasser']
        parcel_names_in_glasser = output_indiv['parcel_names_in_glasser']

        U_individual = output_indiv['U_individual']  # data only parcellation
        indiv_parcellation = output_indiv['indiv_parcellation']

        U_group = output_indiv['U_group']
        group_parcellation = output_indiv['group_parcellation']

        V = output_indiv['V'].T             # n_parcels x n_conditions

    n_subjects, n_parcels, P = U_individual.shape

    ## for mean across subjects
    U_indiv_mean_across_subjects = np.mean(U_individual, axis=0)

    # computing the correlation of the individualized atlas and the glasser group atlas
    cosine_indiv_glasser = np.zeros((n_parcels,))
    for parI in np.arange(n_parcels):
        U_i = U_indiv_mean_across_subjects[parI, included_vtx_inds_LR]
        U_g = U_group[parI, included_vtx_inds_LR]
        cosine_indiv_glasser[parI] = 1 - pdist(np.array([U_i, U_g]), metric='cosine')[0]


    plt.figure()
    for parI in np.arange(n_parcels):
        g = 181 * np.ones(group_parcellation.shape).astype(int)
        g[group_parcellation == labels_in_glasser[parI]] = labels_in_glasser[parI]
        [label_L, label_R] = surf_from_cifti(atlas.data_to_cifti(g.reshape(1, -1)))

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
    resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}')
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    output = dict()
    PKL_output = os.path.join(resultsPath, f'output_{ROI}.pkl')

    # count the number of parcels for each individual
    n_parcels_per_indiv = [len(np.unique(x)) for x in indiv_parcellation[:, included_vtx_inds_LR]]

    # count the number of vertices per parcel
    n_vertices_per_parcel = np.zeros((n_subjects, n_parcels))
    for subjI in np.arange(n_subjects):
        vals, counts = np.unique(indiv_parcellation[subjI, included_vtx_inds_LR], return_counts=True)
        n_vertices_per_parcel[subjI] = counts

    # load V compute similarity between each pair of parcels
    V_simmats = np.corrcoef(V)

    output['V'] = V
    output['V_simmats'] = V_simmats
    output['n_parcels_per_indiv'] = n_parcels_per_indiv
    output['n_vertices_per_parcel'] = n_vertices_per_parcel
    output['cosine_indiv_glasser'] = cosine_indiv_glasser

    JPG_V_corrmat = os.path.join(resultsPath, f'V_corrmat_{ROI}.jpg')
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(V_simmats, cmap='bwr', vmin=-1, vmax=1)
    ax.set_aspect('equal')
    plt.colorbar(im)
    ax.set_title(f'V corrmat {ROI}')
    ax.set_xlabel('parcels')
    ax.set_ylabel('parcels')

    plt.savefig(JPG_V_corrmat, dpi=400, format='jpg')
    plt.close(fig)

    with open(PKL_output, 'wb') as pf:
        pickle.dump(output, pf)


if __name__=='__main__':
    ROIs = ['PFC', 'visual', 'somatosensory', 'parietal']

    for roi in ROIs:
        run_sanity_check(roi)

