import os, subprocess, pickle
import nibabel as nib
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from py_util_dx.py_utils import setProjectPath


def get_roi_vtx_from_fs32k(ROI):
    """
    Get the indices of vertices associated with all the parlces in the roi from fs32k atlas

    Args:
        ROI: list of parcel names or a string which has to be in ['PFC', 'visual', 'parietal', 'somatosensory']

    Returns:

    """
    projectPath, mainResultsPath = setProjectPath()
    surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')
    temp_dir = os.path.join(surface_helpers_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if isinstance(ROI, str):
        PKL_output = os.path.join(surface_helpers_dir, f'roi_vtx_fs32k_{ROI}.pkl')
        if os.path.exists(PKL_output):
            with open(PKL_output, 'rb') as pf:
                output = pickle.load(pf)
                included_vtx_inds_LR = output['included_vtx_inds_LR']
                included_vtx_inds_L = output['included_vtx_inds_L']
                included_vtx_inds_R = output['included_vtx_inds_R']
                excluded_vtx_inds_LR = output['excluded_vtx_inds_LR']
                return included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR
        else:
            parcels = get_roi_pacels(ROI)
    else:
        parcels = ROI

    # load cortical parcellation from label.gii file
    glasser_L = os.path.join(surface_helpers_dir, 'glasser.L.label.gii')
    glasser_R = os.path.join(surface_helpers_dir, 'glasser.R.label.gii')

    # Get the atlas
    atlas_str = 'fs32k'
    atlas, ainf = am.get_atlas(atlas_str)

    gii_files = []
    for hemi in ['L', 'R']:
        if hemi == 'L':
            glasser_label = glasser_L
            flat_shape = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.flat.surf.gii')
        else:
            glasser_label = glasser_R
            flat_shape = os.path.join(surface_helpers_dir, 'fs_LR.32k.R.flat.surf.gii')

        meta = nib.load(flat_shape).meta
        out_gii_file = os.path.join(temp_dir, f'roi_{hemi}.func.gii')
        roi_data = []
        for roi in parcels:
            hemi_roi = f'{hemi}_{roi}'
            out_label = os.path.join(temp_dir, f'{hemi_roi}.func.gii')
            wb_cmd = f'wb_command -gifti-label-to-roi {glasser_label} {out_label} -name {hemi_roi}_ROI'
            subprocess.run(wb_cmd, shell=True)
            roi_data.append(nib.load(out_label).agg_data())
        roi_data = np.array(roi_data).sum(axis=0)
        out_data = nib.gifti.gifti.GiftiImage(meta=meta)
        out_data.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data=roi_data))
        nib.save(out_data, out_gii_file)
        gii_files.append(out_gii_file)

    # roi L, R hemispheres
    label_vec, labels = atlas.get_parcel(gii_files)

    # only keep vertices that are within selected ROIs
    included_vtx_inds_LR = np.where(label_vec > 0)[0]
    included_vtx_inds_L = np.where(label_vec == 1)[0]
    included_vtx_inds_R = np.where(label_vec == 2)[0]
    excluded_vtx_inds_LR = np.where(label_vec == 0)[0]

    if isinstance(ROI, str):
        output = {}
        output['included_vtx_inds_LR'] = included_vtx_inds_LR
        output['included_vtx_inds_L'] = included_vtx_inds_L
        output['included_vtx_inds_R'] = included_vtx_inds_R
        output['excluded_vtx_inds_LR'] = excluded_vtx_inds_LR

        with open(PKL_output, 'wb') as pf:
            pickle.dump(output, pf)

    return included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR


def get_roi_pacels(roi):
    if roi == 'PFC':
        parcels = ['OFC', '10pp', '10r', '8C', 's6-8', '25', 'p24', 'p47r', '46', 'a10p', '10d', '9m', '8Av', 'IFJp', '10v', '13l', '45', 'i6-8', '9-46d', 'IFJa', '47s', 'SFL', 'a24', 'IFSp', '47m', '9p', '9a', 'pOFC', '8Ad', '11l', 'IFSa', 'a9-46v', '44', 'a47r', '55b', '47l', 's32', 'p9-46v', '8BM', 'p10p', '8BL', 'p32', 'a32pr', 'd32']    # PNAS paper
    elif roi == 'visual':
        parcels = ['V1', 'V2', 'V3', 'V4']
    elif roi == 'parietal':
        parcels = ['7AL', '7Am', '7Pm', '7PL', 'MIP', 'VIP', '7PC', 'LIPv', 'AIP', 'LIPd']
    elif roi == 'somatosensory':
        parcels = ['4', '3a', '3b', '1', '2']
    elif roi == 'whole_cortex':
        dict_parcel_labels = get_glasser_labels()
        parcels = list(dict_parcel_labels.keys())
    else:
        print('undefined ROI')
        parcels = []

    return parcels


def get_glasser_labels():
    """
    Getting the labels for the glasser parcellation

    Returns:
        dict_parcel_indices (dict)
        {parcel_name: label from 1-180}
    """

    projectPath, mainResultsPath = setProjectPath()
    surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

    # label table: which can be exported using wb_command -label-export-table glasser_L
    TXT_label_L = os.path.join(surface_helpers_dir, 'glasser.L.label.txt')
    TXT_label_R = os.path.join(surface_helpers_dir, 'glasser.R.label.txt')

    label_table = pd.read_csv(TXT_label_L, header=None)
    all_parcels = [label_table.loc[k, :].to_string().split()[1].replace('_ROI', '').replace('L_', '') for k in np.arange(len(label_table)) if np.mod(k, 2) == 0]
    all_indices = [int(label_table.loc[k, :].to_string().split()[1]) for k in np.arange(len(label_table)) if np.mod(k, 2) == 1]
    dict_parcel_indices = {all_parcels[i]: all_indices[i] for i in np.arange(len(all_parcels))}

    return dict_parcel_indices


def convert_prob_atlas_to_absolute_labels(U, labels_in_glasser, excluded_vtx_inds_LR, inds_U_zero, over_label=181):
    """
    This function converts a probabilistic atlas to a hard parcellation, where each vertex is assigned a label, using the absolute label corresponding to the glasser parcellation

    Args:
        U:                  np.ndarray (n_parcels x 59518 or n_subjects x n_parcels x 59518)
                the probabilistic parcellation
        labels_in_glasser:  np.ndarray
                original labels in the glasser parcellation for the parcels of interest
        excluded_vtx_inds_LR: np.ndarray
                specifying the indices of vertices that fall out of the ROI
        inds_U_zero: np.ndarray (boolean)
                indicating where the 0's are out of the 59518 vertices. 0's can only be part of the excluded_vtx_inds_LR
        over_label: int
                the label for out of range vertices. for glasser, it's 181. for schaefer100, it's 101
    Returns:
        U_labels:           np.ndarray (n_subjects x 59518)
    """

    if U.ndim == 2:
        n_parcels, n_vertices = U.shape
        n_subjects = 1
        U = np.reshape(U, (1, n_parcels, n_vertices))
    elif U.ndim == 3:
        n_subjects, n_parcels, n_vertices = U.shape
    else:
        raise(NameError('U can only be 2d or 3d!'))

    # relative_labels = np.argmax(U, axis=1)
    # relative_labels[:, excluded_vtx_inds_LR] = 181
    # U_labels = 181 * np.ones((n_subjects, n_vertices)).astype(int)
    #
    # for subjI in np.arange(n_subjects):
    #     unique_relative_labels = np.unique(relative_labels[subjI])
    #     for i, n in enumerate(unique_relative_labels):
    #         if n == 181:
    #             continue
    #         U_labels[subjI, relative_labels[subjI]==n] = labels_in_glasser[i]

    relative_labels = np.argmax(U, axis=1)
    U_labels = []

    for subjI in np.arange(n_subjects):
        U_labels.append(labels_in_glasser[relative_labels[subjI]])

    U_labels = np.array(U_labels)
    # U_labels[:, excluded_vtx_inds_LR] = 181
    U_labels[:, excluded_vtx_inds_LR] = over_label
    U_labels[:, inds_U_zero] = 0

    return U_labels
