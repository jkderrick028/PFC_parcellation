import os.path, pickle, scipy
import nibabel as nib
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from py_util_dx.data_utils import get_roi_pacels, get_glasser_labels, get_roi_vtx_from_fs32k
from scipy.stats import ttest_ind, ttest_1samp
from evaluations import *
from Functional_Fusion.reliability import flat2ndarray


"""
This script computes DCBC using individualized parcellation for PFC. 

modified: 2025.06.19
"""


def run_parcellation_evaluation(ROI):
    """
    Evaluate the group and individualized parcellation by computing the prediction error and DCBC
    Args:
        ROI: str
            'PFC', 'visual', 'somatosensory', 'parietal'
    """

    ## defining paths
    projectPath, mainResultsPath = setProjectPath()

    dataset_name = 'MDTB' # or Demand

    # atlas_name = 'glasser'
    # atlas_name = 'schaefer100'
    atlas_name = 'yeo17'

    surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

    resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}', atlas_name)
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    cv = True  # we use cross-validated DCBC
    output = dict()
    if cv:
        PKL_output = os.path.join(resultsPath, f'output_{ROI}_cv.pkl')
    else:
        PKL_output = os.path.join(resultsPath, f'output_{ROI}.pkl')

    ## loading MDTB data
    PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_CondAll_ses-s2.pkl')
    with open(PKL_data, 'rb') as pf:
        original_data = pickle.load(pf)
        X_individuals = original_data['X_individuals']

    # fill nans with 0
    X_individuals[np.isnan(X_individuals)] = 0
    n_subjects = X_individuals.shape[0]

    ## loading individualized parcellation and glasser group parcellation
    # included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(ROI)

    PKL_individualized_parcellation = os.path.join(projectPath, 'results', 'START_A3_bayes_parcellation', f'{dataset_name}', atlas_name, f'output_{ROI}.pkl')
    with open(PKL_individualized_parcellation, 'rb') as pf:
        output_indiv = pickle.load(pf)

        U_individual = output_indiv['U_individual']         # data only parcellation
        indiv_parcellation = output_indiv['indiv_parcellation']

        U_group = output_indiv['U_group']
        group_parcellation = output_indiv['group_parcellation']

        included_vtx_inds_LR = output_indiv['included_vtx_inds_LR']
        included_vtx_inds_L = output_indiv['included_vtx_inds_L']

    U_group = np.tile(U_group, (n_subjects, 1, 1))

    ## prediction error with leave-one-subject-out cross-validation
    cosine_distances_group = prediction_error_cv(U_group[:, :, included_vtx_inds_LR], X_individuals[:, :, included_vtx_inds_LR])
    cosine_distances_indiv = prediction_error_cv(U_individual[:, :, included_vtx_inds_LR], X_individuals[:, :, included_vtx_inds_LR])

    x_group = np.ones(len(cosine_distances_group))
    x_indiv = 2*np.ones(len(cosine_distances_indiv))
    x_cosine_dist = np.concatenate([x_group, x_indiv])
    cosine_dist_concat = np.concatenate([cosine_distances_group, cosine_distances_indiv])
    ttest_result = ttest_ind(cosine_distances_group, cosine_distances_indiv, alternative='two-sided')

    plt.figure()
    plt.scatter(x_cosine_dist, cosine_dist_concat)
    plt.xticks([1, 2], labels=['group', 'indiv'])
    plt.xlabel('atlas type')
    plt.ylabel('cosine dist')
    plt.title(f'prediction error {ROI}, p={ttest_result.pvalue}')
    JPG_fig = os.path.join(resultsPath, f'prediction_error_{ROI}.jpg')
    plt.savefig(JPG_fig, dpi=500, format='jpg')

    ## DCBC using left hemisphere only
    MAT_dist = os.path.join(projectPath, 'code', 'DCBC', 'distanceMatrix', 'distAvrg_sp.mat')
    spatialMat = scipy.io.loadmat(MAT_dist)['avrgDs'].toarray()

    glasser_L = os.path.join(surface_helpers_dir, f'glasser.L.label.gii')

    # extract the first data array as the parcels
    # make sure that the input parcels are of shape (N,)
    gii_file = nib.load(glasser_L)

    parcels_inds = gii_file.darrays[0].data
    parcels_ROI = get_roi_pacels(ROI)
    dict_parcel_indices = get_glasser_labels()  # {parcel: label}
    indices_ROI = [dict_parcel_indices[k] for k in parcels_ROI]

    # for alternative atlas
    if atlas_name != 'glasser':
        alternative_atlas_L = os.path.join(surface_helpers_dir, f'{atlas_name}.L.label.gii')
        parcels_inds_alt = nib.load(alternative_atlas_L).darrays[0].data

    # get all the vertices that are in the ROI list
    vertex_label_ROI, vertex_ind_ROI = [], []
    for i, label in enumerate(parcels_inds):
        if label in indices_ROI:
            # vertex_label_ROI.append(label)
            # vertex_ind_ROI.append(i)

            if atlas_name == 'yeo17':
                if ROI == 'PFC':
                    if parcels_inds_alt[i] in [0, 6, 11]:
                        continue
                elif ROI == 'somatosensory':
                    if parcels_inds_alt[i] in [0, 7]:
                        continue
                elif ROI == 'visual':
                    if parcels_inds_alt[i] in [0]:
                        continue
                elif ROI == 'parietal':
                    if parcels_inds_alt[i] in [0, 16]:
                        continue
                vertex_label_ROI.append(parcels_inds_alt[i])
                vertex_ind_ROI.append(i)

            elif atlas_name == 'glasser':
                vertex_label_ROI.append(label)
                vertex_ind_ROI.append(i)

            elif atlas_name == 'schaefer100':
                if parcels_inds_alt[i] in [0]:
                        continue
                vertex_label_ROI.append(parcels_inds_alt[i])
                vertex_ind_ROI.append(i)

    vertex_label_ROI = np.array(vertex_label_ROI)
    vertex_ind_ROI = np.array(vertex_ind_ROI)

    spatialMat = spatialMat[vertex_ind_ROI, :]
    spatialMat = spatialMat[:, vertex_ind_ROI]

    # Create a DCBC evaluation object of the desired evaluation parameters(left hemisphere)
    if cv:
        ## loading MDTB data
        PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_CondHalf_ses-s2.pkl')
        with open(PKL_data, 'rb') as pf:
            original_data = pickle.load(pf)
            X_individuals = original_data['X_individuals']
            info_individuals = original_data['info_individuals']
            dataset_obj_individuals = original_data['dataset_obj_individuals']

        # fill nans with 0
        X_individuals[np.isnan(X_individuals)] = 0
        n_subjects = X_individuals.shape[0]

        part_vec = list(info_individuals['half'])
        cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])
        data = flat2ndarray(X_individuals[:, :, included_vtx_inds_L], part_vec, cond_vec)

    else:
        X_individuals[np.isnan(X_individuals)] = 0
        n_subjects = X_individuals.shape[0]
        data = X_individuals[:, :, included_vtx_inds_L]

    U_indiv_label = indiv_parcellation[:, included_vtx_inds_L]
    U_group_label = np.tile(group_parcellation, (n_subjects, 1))[:, included_vtx_inds_L]

    output_dcbc_group = compute_dcbc_indiv(U_group_label, data, spatialMat, cv=cv)
    output_dcbc_indiv = compute_dcbc_indiv(U_indiv_label, data, spatialMat, cv=cv)

    ## making results figures for group atlas
    within_corrs = output_dcbc_group['within_corrs']
    between_corrs = output_dcbc_group['between_corrs']
    dcbc_group = output_dcbc_group['dcbc']
    if cv:
        JPG_fig = os.path.join(resultsPath, f'DCBC_group_{ROI}_cv.jpg')
    else:
        JPG_fig = os.path.join(resultsPath, f'DCBC_group_{ROI}.jpg')


    def summarize_dcbc(within_corrs, between_corrs, dcbc, fig_path, atlas_type):
        """
        summarizing dcbc results
        Args:
            within_corrs:
            between_corrs:
            dcbc:

        Returns:

        """

        within_corrs_mean = np.mean(within_corrs, axis=0)
        within_corrs_ste = np.std(within_corrs, axis=0) / np.sqrt(n_subjects)
        between_corrs_mean = np.mean(between_corrs, axis=0)
        between_corrs_ste = np.std(between_corrs, axis=0) / np.sqrt(n_subjects)

        ## testing weather dcbc is significantly higher than 0
        dcbc = np.array(dcbc)
        ttest_result = ttest_1samp(dcbc, 0, alternative='greater')
        significance_level = 0.05
        is_significant = ttest_result.pvalue < significance_level

        fig, ax = plt.subplots(1, 1)
        ax.errorbar(5+np.arange(0, 35, step=5), within_corrs_mean, yerr=within_corrs_ste)
        ax.errorbar(5+np.arange(0, 35, step=5), between_corrs_mean, yerr=between_corrs_ste)
        ax.set_xlabel('distance (mm)')
        ax.set_ylabel('vertex-to-vertex correlation')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(['within', 'between'], frameon=False)
        if is_significant:
            ax.set_title(f'{ROI} {atlas_type}, dcbc significant')
        else:
            ax.set_title(f'{ROI} {atlas_type}, dcbc not significant')

        plt.savefig(fig_path, dpi=500, format='jpg')


    summarize_dcbc(within_corrs, between_corrs, dcbc_group, JPG_fig, 'group')

    ## making results figures for individualized atlas
    within_corrs = output_dcbc_indiv['within_corrs']
    between_corrs = output_dcbc_indiv['between_corrs']
    dcbc_indiv = output_dcbc_indiv['dcbc']
    if cv:
        JPG_fig = os.path.join(resultsPath, f'DCBC_indiv_{ROI}_cv.jpg')
    else:
        JPG_fig = os.path.join(resultsPath, f'DCBC_indiv_{ROI}.jpg')

    summarize_dcbc(within_corrs, between_corrs, dcbc_indiv, JPG_fig, 'indiv')

    x_group = np.ones(len(dcbc_group))
    x_indiv = 2*np.ones(len(dcbc_indiv))
    x_dcbc = np.concatenate([x_group, x_indiv])
    dcbc_concat = np.concatenate([dcbc_group, dcbc_indiv])
    plt.figure()
    plt.scatter(x_dcbc, dcbc_concat)
    plt.xticks([1, 2], labels=['group', 'indiv'])
    plt.xlabel('atlas type')
    plt.ylabel('dcbc')
    plt.title(f'DCBC using {atlas_name} and indiv atlas')
    if cv:
        JPG_fig = os.path.join(resultsPath, f'DCBC_scatter_{ROI}_cv.jpg')
    else:
        JPG_fig = os.path.join(resultsPath, f'DCBC_scatter_{ROI}.jpg')
    plt.savefig(JPG_fig, dpi=500, format='jpg')

    output['output_dcbc_group'] = output_dcbc_group
    output['output_dcbc_indiv'] = output_dcbc_indiv
    output['cosine_distances_group'] = cosine_distances_group
    output['cosine_distances_indiv'] = cosine_distances_indiv

    with open(PKL_output, 'wb') as pk:
        pickle.dump(output, pk)



if __name__=='__main__':
    ROIs = ['PFC', 'visual', 'somatosensory', 'parietal']

    for roi in ROIs:
        run_parcellation_evaluation(roi)
