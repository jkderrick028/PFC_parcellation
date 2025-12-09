import numpy as np
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.util as ut
from py_util_dx.py_utils import setProjectPath
import os, pickle
from nitools.cifti import surf_from_cifti
import torch
from visualizations import plot_flatmap_labels
from py_util_dx.data_utils import get_roi_pacels, get_roi_vtx_from_fs32k, convert_prob_atlas_to_absolute_labels


def run_bayes_parcellation(ROI):
    """
    Individualized parcellation using HBP framework for ROI
    Args:
        ROI: str
            'PFC', 'visual', 'somatosensory', 'parietal'
    """

    projectPath, mainResultsPath = setProjectPath()

    dataset_name = 'MDTB' # or Demand

    # atlas_name = 'yeo17'
    atlas_name = 'schaefer100'
    # atlas_name = 'glasser'

    strength = 20.0

    resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}', atlas_name)
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

    PKL_output = os.path.join(resultsPath, f'output_{ROI}.pkl')
    output = {}

    ## loading MDTB data
    PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_CondAll_ses-s1.pkl')
    with open(PKL_data, 'rb') as pf:
        original_data = pickle.load(pf)
        data = original_data['X_individuals']
        info_individuals = original_data['info_individuals']
        dataset_obj_individuals = original_data['dataset_obj_individuals']

    cond_vec = np.array(list(info_individuals[dataset_obj_individuals.cond_ind]))
    # part_vec = np.array(list(info_individuals[dataset_obj_individuals.part_ind]))
    part_vec = np.ones(len(cond_vec)).astype(int)

    ## loading group atlas
    atlas_str = 'fs32k'
    atlas, ainf = am.get_atlas(atlas_str)

    # Sample the probabilistic atlas at the specific atlas grayordinates
    # atlas_fname = [os.path.join(surface_helpers_dir, 'glasser.L.label.gii'), os.path.join(surface_helpers_dir, 'glasser.R.label.gii')]    # glasser atlas
    atlas_fname = [os.path.join(surface_helpers_dir, f'{atlas_name}.L.label.gii'), os.path.join(surface_helpers_dir, f'{atlas_name}.R.label.gii')]
    U = atlas.read_data(atlas_fname)

    n_vertices_whole_cortex = len(U)
    inds_U_zero = U == 0

    ## dealing with PFC mask
    if ROI == 'whole_cortex':
        included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k('whole_cortex')
    else:
        included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(ROI)

    # for glasser atlas, included_vtx_inds_LR does not include 0 any more. However, for other atlases, such as the yeo17 atlas, there could still be 0 label.
    if atlas_name == 'yeo17':
        if ROI == 'somatosensory':
            vtx_0_label_LR = included_vtx_inds_LR[np.isin(U[included_vtx_inds_LR], [0, 7])]
            vtx_0_label_L = included_vtx_inds_L[np.isin(U[included_vtx_inds_L], [0, 7])]
            vtx_0_label_R = included_vtx_inds_R[np.isin(U[included_vtx_inds_R], [0, 7])]
        elif ROI == 'PFC':
            vtx_0_label_LR = included_vtx_inds_LR[np.isin(U[included_vtx_inds_LR], [0, 6, 11])]
            vtx_0_label_L = included_vtx_inds_L[np.isin(U[included_vtx_inds_L], [0, 6, 11])]
            vtx_0_label_R = included_vtx_inds_R[np.isin(U[included_vtx_inds_R], [0, 6, 11])]
        elif ROI == 'parietal':
            vtx_0_label_LR = included_vtx_inds_LR[np.isin(U[included_vtx_inds_LR], [0, 16])]
            vtx_0_label_L = included_vtx_inds_L[np.isin(U[included_vtx_inds_L], [0, 16])]
            vtx_0_label_R = included_vtx_inds_R[np.isin(U[included_vtx_inds_R], [0, 16])]
        else:
            vtx_0_label_LR = included_vtx_inds_LR[np.isin(U[included_vtx_inds_LR], [0])]
            vtx_0_label_L = included_vtx_inds_L[np.isin(U[included_vtx_inds_L], [0])]
            vtx_0_label_R = included_vtx_inds_R[np.isin(U[included_vtx_inds_R], [0])]

        included_vtx_inds_LR = included_vtx_inds_LR[~np.isin(included_vtx_inds_LR, vtx_0_label_LR)]
        included_vtx_inds_L = included_vtx_inds_L[~np.isin(included_vtx_inds_L, vtx_0_label_L)]
        included_vtx_inds_R = included_vtx_inds_R[~np.isin(included_vtx_inds_R, vtx_0_label_R)]
        excluded_vtx_inds_LR = np.concatenate([excluded_vtx_inds_LR, vtx_0_label_LR])

    elif atlas_name == 'schaefer100':
        vtx_0_label_LR = included_vtx_inds_LR[np.isin(U[included_vtx_inds_LR], [0])]
        vtx_0_label_L = included_vtx_inds_L[np.isin(U[included_vtx_inds_L], [0])]
        vtx_0_label_R = included_vtx_inds_R[np.isin(U[included_vtx_inds_R], [0])]

        included_vtx_inds_LR = included_vtx_inds_LR[~np.isin(included_vtx_inds_LR, vtx_0_label_LR)]
        included_vtx_inds_L = included_vtx_inds_L[~np.isin(included_vtx_inds_L, vtx_0_label_L)]
        included_vtx_inds_R = included_vtx_inds_R[~np.isin(included_vtx_inds_R, vtx_0_label_R)]
        excluded_vtx_inds_LR = np.concatenate([excluded_vtx_inds_LR, vtx_0_label_LR])


    labels_relative = U[included_vtx_inds_LR]
    data = data[:, :, included_vtx_inds_LR]

    n_subjects = data.shape[0]

    ## converting the hard parcellation into a probabilistic one
    labels_in_group, labels_relative = np.unique(labels_relative, return_inverse=True)    # labels_in_group is a list of labels of parcels of interest in glasser parcellation, starting from 1
    if atlas_name == 'glasser':
        parcel_names = get_roi_pacels('whole_cortex')
        parcel_names_in_group = [parcel_names[k - 1] for k in labels_in_group]
    elif atlas_name in ['yeo17', 'schaefer100']:
        parcel_names = [f'{x}' for x in labels_in_group]
        parcel_names_in_group = parcel_names

    ## K is the number of parcels
    K = len(labels_in_group)

    logpi = ar.expand_mn_1d(labels_relative, K) * strength
    U_roi = torch.softmax(logpi, dim=0)

    # Build the arrangement model - the parameters are the log-probabilities of the atlas
    ar_model = ar.build_arrangement_model(U_roi, prior_type='logpi', atlas=atlas)
    # ar_model = ar.build_arrangement_model(U_roi, prior_type='logpi', atlas=atlas, sym_type='asym' if atlas_name in ['schaefer100'] else 'sym')

    # fit the emission model to the data
    # Make a design matrix
    X= ut.indicator(cond_vec)
    # Build an emission model
    em_model = em.MixVMF(K=K, P=U_roi.shape[1], X=X, part_vec=part_vec)

    # Build the full model: The emission models are passed as a list, as usually we have multiple data sets
    M = fm.FullMultiModel(ar_model, [em_model])
    # Attach the data to the model - this is done for speed
    # The data is passed as a list with on element per data set
    M.initialize([data])

    # Now we can run the EM algorithm
    M, ll, _, U_indiv = M.fit_em(iter=1000, tol=0.01, fit_arrangement=False,fit_emission=True,first_evidence=False)

    # get the data only parcellation
    emloglik  = M.emissions[0].Estep()
    Uhat_data = torch.softmax(emloglik, dim=1).numpy()

    # printing kappa
    print(f'kappa: {M.emissions[0].kappa}')

    ## restoring U to the original shape (containing all vertices in the cortex)
    ## U_group: the softened glasser parcellation, n_parcels x n_vertices in whole cortex
    U_group_restore = np.zeros((K, n_vertices_whole_cortex))
    U_group_restore[:, included_vtx_inds_LR] = U_roi
    U_group = U_group_restore.copy()

    ## group parcellation
    group_parcellation = U.copy()
    # group_parcellation[excluded_vtx_inds_LR] = 181
    group_parcellation[excluded_vtx_inds_LR] = np.max(group_parcellation) + 1
    group_parcellation[inds_U_zero] = 0

    ## U_individual: the data likelihood, n_subjects x n_parcels x n_vertices in whole cortxe
    U_individual_restore = np.zeros((n_subjects, K, n_vertices_whole_cortex))
    U_individual_restore[:, :, included_vtx_inds_LR] = Uhat_data
    U_individual = U_individual_restore.copy()

    ## derive the individualized parcellation from the U_individual using winnder-take-all
    ## indiv_parcellation (n_subjects x n_vertices in whole cortex)
    indiv_parcellation = convert_prob_atlas_to_absolute_labels(U_individual, labels_in_group, excluded_vtx_inds_LR, inds_U_zero, over_label=np.max(group_parcellation))

    # saving U and U_indiv
    output['U_group'] = U_group
    output['group_parcellation'] = group_parcellation
    output['U_individual'] = U_individual
    output['indiv_parcellation'] = indiv_parcellation
    output['ll'] = ll
    output['labels_in_group'] = labels_in_group
    output['parcel_names_in_group'] = parcel_names_in_group
    output['V'] = M.emissions[0].V.numpy()
    output['included_vtx_inds_LR'] = included_vtx_inds_LR
    output['included_vtx_inds_L'] = included_vtx_inds_L
    output['included_vtx_inds_R'] = included_vtx_inds_R
    output['excluded_vtx_inds_LR'] = excluded_vtx_inds_LR
    # output['M'] = M
    # output['theta'] = theta

    with open(PKL_output, 'wb') as pf:
        pickle.dump(output, pf)


    ## plotting group atlas
    plt.figure()
    [label_L, label_R] = surf_from_cifti(atlas.data_to_cifti(group_parcellation.reshape(1, -1)))
    plt.subplot(1, 2, 1)
    plot_flatmap_labels(label_L, 'L', atlas=atlas_name)

    plt.subplot(1, 2, 2)
    plot_flatmap_labels(label_R, 'R', atlas=atlas_name)

    plt.suptitle('group')
    plt.tight_layout()
    JPG_fig = os.path.join(resultsPath, f'{ROI}_group_parcellations.jpg')
    plt.savefig(JPG_fig, format='jpg', dpi=400)

    # plot individualized parcellations for 3 subjects
    plt.figure()
    for i,s in enumerate([6, 9, 12]):
        plt.clf()
        [label_L, label_R] = surf_from_cifti(atlas.data_to_cifti(indiv_parcellation[i].reshape(1, -1)))
        plt.subplot(1, 2, 1)
        plot_flatmap_labels(label_L, 'L', atlas=atlas_name)
        plt.subplot(1, 2, 2)
        plot_flatmap_labels(label_R, 'R', atlas=atlas_name)
        plt.suptitle(f'subject {s}')
        plt.tight_layout()

        JPG_fig = os.path.join(resultsPath, f'{ROI}_indiv_parcellations_subject_{s}.jpg')
        plt.savefig(JPG_fig, format='jpg', dpi=400)

    # inspect model training
    plt.figure(figsize=(5,5))
    plt.plot(ll)

    JPG_fig = os.path.join(resultsPath, f'll_training_{ROI}.jpg')
    plt.savefig(JPG_fig, format='jpg', dpi=400)

    pass


if __name__=='__main__':
    ROIs = ['PFC', 'parietal', 'visual', 'somatosensory']

    for roi in ROIs:
        run_bayes_parcellation(roi)
