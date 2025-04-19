import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds


def flat2ndarray(flat_data, part_vec, cond_vec):
    """
    convert flat data (n_subjects x n_trials x n_voxels) into a 4d ndarray (n_subjects x n_partitions x n_conditions x n_voxels)

    Args:
        flat_data:
        part_vec:
        cond_vec:

    Returns:
        data

    """

    [n_subjects, n_trials, n_voxels] = flat_data.shape

    unique_partitions = np.unique(part_vec)
    n_partitions = unique_partitions.size

    unique_conditions = np.unique(cond_vec)
    n_conditions = unique_conditions.size

    data = np.zeros((n_subjects, n_partitions, n_conditions, n_voxels))

    for partI in np.arange(n_partitions):
        for condI in np.arange(n_conditions):
            trial_inds = np.where(np.logical_and(cond_vec == unique_conditions[condI], part_vec == unique_partitions[partI]))[0]
            data[:, partI, condI, :] = np.nanmean(flat_data[:, trial_inds, :], axis=1).squeeze()

    return data
