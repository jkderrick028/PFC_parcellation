import os.path, pickle
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from py_util_dx.data_utils import get_roi_vtx_from_fs32k
from Functional_Fusion.reliability import decompose_subj_group, flat2ndarray
from Functional_Fusion_old.dataset import decompose_pattern_into_group_indiv_noise
import SUITPy.flatmap as flatmap
from nitools.cifti import surf_from_cifti

"""
This scripts decomposes the data into group, individual and noise variance components. 
"""


def START_A2_gse_decomposition(dataset_name):

    projectPath, mainResultsPath = setProjectPath()

    np.random.seed(0)

    # dataset_name = 'MDTB' # or Demand

    surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

    resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    output = dict()
    output['whole_cortex'] = {}
    PKL_output = os.path.join(resultsPath, f'{dataset_name}_output.pkl')

    PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_Half.pkl')
    with open(PKL_data, 'rb') as pf:
        original_data = pickle.load(pf)
        X_individuals = original_data['X_individuals']
        info_individuals = original_data['info_individuals']
        dataset_obj_individuals = original_data['dataset_obj_individuals']

    # part_vec = list(info_individuals['half'])
    # # part_vec = [int(x[-1]) for x in part_vec]
    # cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

    # base_dir = '/cifs/diedrichsen/data/FunctionalFusion_new'
    #
    # X_individuals, info_individuals, dataset_obj = ds.get_dataset(base_dir,
    #                                                               dataset=dataset_name,
    #                                                               atlas='fs32k',
    #                                                               sess='all',
    #                                                               type='CondHalf')

    part_vec = list(info_individuals['half'])
    cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

    # fill nans with 0
    X_individuals[np.isnan(X_individuals)] = 0

    ## voxel-wise decomposition for the whole cortex: make a flatmap for the entire cortex (vs/(vs+vg))
    criterion = 'global'
    variances = decompose_subj_group(X_individuals, cond_vec, part_vec, separate='none')
    output['whole_cortex'][criterion] = variances

    criterion = 'voxel_wise'
    variances = decompose_subj_group(X_individuals, cond_vec, part_vec, separate=criterion)
    output['whole_cortex'][criterion] = variances

    criterion = 'condition_wise'
    variances = decompose_subj_group(X_individuals, cond_vec, part_vec, separate=criterion)
    output['whole_cortex'][criterion] = variances


    ## global response pattern decomposition for ROIs: bootstrap resample conditions
    # Get the atlas
    atlas_str = 'fs32k'
    atlas, ainf = am.get_atlas(atlas_str)

    # Read data from two gifti files for left and right hemisphere
    glasser_left = os.path.join(surface_helpers_dir, 'glasser.L.label.gii')
    glasser_right = os.path.join(surface_helpers_dir, 'glasser.R.label.gii')

    flat_surf_L = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.flat.surf.gii')
    flat_surf_R = os.path.join(surface_helpers_dir, 'fs_LR.32k.R.flat.surf.gii')

    ROIs = ['PFC', 'visual', 'somatosensory', 'parietal']
    n_rois = len(ROIs)


    ## whole cortex
    data = flat2ndarray(X_individuals, part_vec, cond_vec)

    n_bootstraps = 100
    n_subjects, n_partitions, n_conditions, n_voxels = data.shape
    boot_conditions = []
    for bootI in np.arange(n_bootstraps):
        conds = np.random.choice(n_conditions, n_conditions, replace=True)
        boot_conditions.append(conds)

    for roiI in np.arange(n_rois):
        output[ROIs[roiI]] = {}
        included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(ROIs[roiI])

        criterion = 'global'
        output[ROIs[roiI]][criterion] = []

        # first get the decomposition results on the original data without bootstrapping conditions
        data_roi = data[:, :, :, included_vtx_inds_LR]
        variances = decompose_pattern_into_group_indiv_noise(data_roi, criterion=criterion)
        output[ROIs[roiI]][criterion].append(variances.flatten())

        for bootI in np.arange(n_bootstraps):
            data_roi = data[:, :, :, included_vtx_inds_LR]
            data_roi = data_roi[:, :, boot_conditions[bootI], :]
            variances = decompose_pattern_into_group_indiv_noise(data_roi, criterion=criterion)
            output[ROIs[roiI]][criterion].append(variances.flatten())


    with open(PKL_output, 'wb') as pk:
        pickle.dump(output, pk)

    ## flatmap visualization
    underlay_L = os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii')
    underlay_R = os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii')
    border_LR = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.border')

    # normalization_method = 'gse'    # gse or gs
    normalization_method = 'gs'

    if normalization_method == 'gse':
        voxel_wise_cortex = np.divide(output['whole_cortex']['voxel_wise'], np.sum(output['whole_cortex']['voxel_wise'], axis=1, keepdims=True))
    else:
        voxel_wise_cortex = np.divide(output['whole_cortex']['voxel_wise'][:, 0:2], np.sum(output['whole_cortex']['voxel_wise'][:, 0:2], axis=1, keepdims=True))

    # v_g
    [v_g_extended_L, v_g_extended_R] = surf_from_cifti(atlas.data_to_cifti(voxel_wise_cortex[:, 0].reshape(1, -1)))

    figI_flatmap = 11
    nHors = 1
    nVers = 2
    fig, axs = plt.subplots(nHors, nVers, figsize=(15, 12), num=figI_flatmap)
    plt.axes(axs[0])
    flatmap.plot(v_g_extended_L.reshape(-1, ), surf=flat_surf_L, underlay=underlay_L, alpha=1, cscale=[0, 0.9], borders=border_LR, frame=None, new_figure=False)
    axs[0].set_title('cortex L')
    plt.axes(axs[1])
    flatmap.plot(v_g_extended_R.reshape(-1, ), surf=flat_surf_R, underlay=underlay_R, alpha=1, cscale=[0, 0.9], borders=border_LR, frame=None, new_figure=False, colorbar=True)
    axs[1].set_title('cortex R')

    # plt.tight_layout()
    plt.suptitle('v_g')

    PS_variance = os.path.join(resultsPath, f'{dataset_name}_flatmap_v_g_{normalization_method}.png')
    if os.path.exists(PS_variance):
        os.remove(PS_variance)
    fig.savefig(PS_variance, format='png', dpi=500)

    # v_s
    [v_s_extended_L, v_s_extended_R] = surf_from_cifti(atlas.data_to_cifti(voxel_wise_cortex[:, 1].reshape(1, -1)))
    plt.clf()

    fig, axs = plt.subplots(nHors, nVers, figsize=(15, 12), num=figI_flatmap)
    plt.axes(axs[0])
    flatmap.plot(v_s_extended_L.reshape(-1, ), surf=flat_surf_L, underlay=underlay_L, alpha=1, cscale=[0, 0.9], borders=border_LR, frame=None, new_figure=False)
    axs[0].set_title('cortex L')
    plt.axes(axs[1])
    flatmap.plot(v_s_extended_R.reshape(-1, ), surf=flat_surf_R, underlay=underlay_R, alpha=1, cscale=[0, 0.9], borders=border_LR, frame=None, new_figure=False, colorbar=True)
    axs[1].set_title('cortex R')

    # plt.tight_layout()
    plt.suptitle('v_s')

    PS_variance = os.path.join(resultsPath, f'{dataset_name}_flatmap_v_s_{normalization_method}.png')
    if os.path.exists(PS_variance):
        os.remove(PS_variance)
    fig.savefig(PS_variance, format='png', dpi=500)

    # v_e
    if normalization_method == 'gse':
        [v_e_extended_L, v_e_extended_R] = surf_from_cifti(atlas.data_to_cifti(voxel_wise_cortex[:, 2].reshape(1, -1)))
        plt.clf()

        fig, axs = plt.subplots(nHors, nVers, figsize=(15, 12), num=figI_flatmap)
        plt.axes(axs[0])
        flatmap.plot(v_e_extended_L.reshape(-1, ), surf=flat_surf_L, underlay=underlay_L, alpha=1, cscale=[-0.1, 0.7], borders=border_LR, frame=None, new_figure=False)
        axs[0].set_title('cortex L')
        plt.axes(axs[1])
        flatmap.plot(v_e_extended_R.reshape(-1, ), surf=flat_surf_R, underlay=underlay_R, alpha=1, cscale=[-0.1, 0.7], borders=border_LR, frame=None, new_figure=False, colorbar=True)
        axs[1].set_title('cortex R')

        # plt.tight_layout()
        plt.suptitle('v_e')

        PS_variance = os.path.join(resultsPath, f'{dataset_name}_flatmap_v_e_{normalization_method}.png')
        if os.path.exists(PS_variance):
            os.remove(PS_variance)
        fig.savefig(PS_variance, format='png', dpi=500)
    plt.show()



if __name__=='__main__':
    datasets = ['MDTB', 'HCPur100', 'Nishimoto']

    for dataset in datasets:
        START_A2_gse_decomposition(dataset_name=dataset)
