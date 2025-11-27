import numpy as np
from scipy.spatial import distance
import DCBC.dcbc as DCBC
from DCBC.utilities import compute_var_cov
import torch as pt
import scipy as sp
from scipy.optimize import curve_fit


# def spatial_ACF_cv(maxDist=35, binWidth=1, func=None, dist=None):
#     """
#     cross-validated spatial ACF
#
#     Args:
#         maxDist: The maximum distance for vertices pairs, default 35 mm
#         binWidth: The spatial binning width in mm, default 1 mm
#         func: the functional data for evaluating, shape (N, P),
#               N - the dimensionality of underlying data, i.e. the number
#               of task contrasts or the number of resting-state networks
#               P - the number of brain voxels / vertices
#               Note: if cv is True, func is of shape (R, N, P) where R
#               is the number of runs or partitions
#         dist: the pairwise distance matrix between P brain locations. It
#               can be a dense matrix or sparse tensor
#
#     Returns:
#         D: a dictionary contains necessary information for DCBC analysis
#     """
#     numBins = int(np.floor(maxDist / binWidth))
#     cov, var = compute_var_cov(func, backend='numpy', cv=True)
#
#     # remove the nan value and medial wall from dist file
#     row, col, distance = sp.sparse.find(dist)
#
#     nums, corrs = [], []
#     # at distance 0
#     nums.append(len(np.diagonal(cov)))
#     # corrs.append(np.nanmean(np.diagonal(cov)) / np.nanmean(np.diagonal(var)))
#     corrs.append(np.nanmean(np.diagonal(cov) / np.diagonal(var)))
#
#     for i in range(numBins):
#         inBin = np.where((distance > i * binWidth) & (distance <= (i + 1) * binWidth))[0]
#
#         # retrieve and append the number of vertices for within/between in current bin
#         nums.append(len(inBin))
#
#         # Compute and append averaged within- and between-parcel correlations in current bin
#         # this_corr = (np.nanmean(cov[row[inBin], col[inBin]])
#         #              / np.nanmean(var[row[inBin], col[inBin]]))
#
#         this_corr = np.nanmean(cov[row[inBin], col[inBin]] / var[row[inBin], col[inBin]])
#
#         corrs.append(this_corr)
#
#         del inBin
#
#     nums = np.array(nums)
#     corrs = np.array(corrs)
#     dists = np.arange(0, maxDist+0.5, binWidth)
#
#     D = {
#         "binWidth": binWidth,
#         "maxDist": maxDist,
#         "nums": nums,
#         "corrs": corrs,
#         "dists": dists
#     }
#
#     return D


def laplacian_pdf(x, b):

    return np.exp(-x/b)


def spatial_ACF_cv(maxDist=35, binWidth=1, func=None, dist=None):
    """
    cross-validated spatial ACF

    Args:
        maxDist: The maximum distance for vertices pairs, default 35 mm
        binWidth: The spatial binning width in mm, default 1 mm
        func: the functional data for evaluating, shape (N, P),
              N - the dimensionality of underlying data, i.e. the number
              of task contrasts or the number of resting-state networks
              P - the number of brain voxels / vertices
              Note: if cv is True, func is of shape (R, N, P) where R
              is the number of runs or partitions
        dist: the pairwise distance matrix between P brain locations. It
              can be a dense matrix or sparse tensor

    Returns:
        D: a dictionary contains necessary information for DCBC analysis
    """
    numBins = int(np.floor(maxDist / binWidth))
    cov, var = compute_var_cov(func, backend='numpy', cv=True)

    # remove the nan value and medial wall from dist file
    row, col, distance = sp.sparse.find(dist)

    nums, corrs = [], []

    # at distance 0
    nums.append(len(np.diagonal(cov)))
    corrs.append(np.nanmean(np.diagonal(cov)) / np.nanmean(np.diagonal(cov)))

    for i in range(numBins):
        inBin = np.where((distance > i * binWidth) & (distance <= (i + 1) * binWidth))[0]

        # retrieve and append the number of vertices for within/between in current bin
        nums.append(len(inBin))

        # Compute and append averaged within- and between-parcel correlations in current bin
        this_corr = (np.nanmean(cov[row[inBin], col[inBin]])
                     / np.nanmean(var[row[inBin], col[inBin]]))

        corrs.append(this_corr)

        del inBin

    nums = np.array(nums)
    corrs = np.array(corrs)
    dists = np.arange(0, maxDist+0.5, binWidth)

    D = {
        "binWidth": binWidth,
        "maxDist": maxDist,
        "nums": nums,
        "corrs": corrs,
        "dists": dists
    }

    return D



def spatial_ACF_per_voxel_cv(maxDist=35, binWidth=1, func=None, dist=None):
    """
    cross-validated spatial ACF for each voxel, estimate a FWHM for each voxel

    Args:
        maxDist: The maximum distance for vertices pairs, default 35 mm
        binWidth: The spatial binning width in mm, default 1 mm
        func: the functional data for evaluating, shape (N, P),
              N - the dimensionality of underlying data, i.e. the number
              of task contrasts or the number of resting-state networks
              P - the number of brain voxels / vertices
              Note: if cv is True, func is of shape (R, N, P) where R
              is the number of runs or partitions
        dist: the pairwise distance matrix between P brain locations. It
              can be a dense matrix or sparse tensor

    Returns:
        D: a dictionary contains necessary information for DCBC analysis
    """
    numBins = int(np.floor(maxDist / binWidth))
    cov, var = compute_var_cov(func, backend='numpy', cv=True)

    P = cov.shape[0]
    corrs = np.zeros((P, numBins+1))
    FWHMs = []
    dists = np.arange(0, maxDist + 0.5, binWidth)

    for p in np.arange(P):
        # at distance 0
        corrs[p, 0] = 1
        for i in np.arange(numBins):
            inBin = np.where((dist[p] > i * binWidth) & (dist[p] <= (i + 1) * binWidth))[0]
            cov_ = cov[p, inBin]
            var_ = var[p, inBin]
            this_corr = np.nanmean(cov_[~np.isnan(var_)]) / np.nanmean(var[p, inBin])
            corrs[p, i+1] = this_corr

        # try:
        #     params, params_cov = curve_fit(laplacian_pdf, dists, corrs[p], nan_policy='omit', maxfev=10000)
        # except:
        #     breakpoint()

        params, params_cov = curve_fit(laplacian_pdf, dists, corrs[p], nan_policy='omit', maxfev=10000)

        FWHMs.append(params[0])

    FWHMs = np.array(FWHMs)
    FWHMs = 2 * FWHMs * np.log(2)

    D = {
        "binWidth": binWidth,
        "maxDist": maxDist,
        "FWHMs": FWHMs,
        "corrs": corrs,
        "dists": dists
    }

    return D


def prediction_error_cv(U_hat, Y_test, type='hard'):
    """
    Evaluating parcellation by computing prediction error with leave-one-subject-out cross-validation
    Args:
        U_hat: np.ndarray (n_subjects x n_parcels x n_vertices)
                individualized parcellation
        Y_test: np.ndarray (n_subjects x n_conditions x n_vertices)
                test data
    Returns:
        cosine distances between predicted Y and actual Y_test

    2025.05.06
    """

    if type == 'hard':
        idx = pt.argmax(pt.tensor(U_hat), dim=1, keepdim=True)
        U = pt.zeros_like(pt.tensor(U_hat)).scatter_(1, idx, 1.).numpy()
    elif type == 'expected':
        U = U_hat.copy()

    n_subjects, n_parcels, n_vertices = U.shape
    cosine_distances = []

    for subjI in np.arange(n_subjects):
        train_s = [s for s in np.arange(n_subjects) if s != subjI]
        Y_train = Y_test[train_s]
        Y_train = Y_train.transpose([0, 2, 1])
        U_train = U[train_s]
        U_train = np.divide(U_train, np.sum(U_train, axis=-1, keepdims=True))
        V = np.matmul(U_train, Y_train)
        V = V.mean(axis=0)

        U_test = np.divide(U[subjI], np.sum(U[subjI], axis=-1, keepdims=True))
        Y_pred = (np.linalg.pinv(U_test) @ V).T
        cos_dist = distance.cosine(Y_pred.flatten(), Y_test[subjI].flatten())
        cosine_distances.append(cos_dist)

    return np.array(cosine_distances)


def compute_dcbc_indiv(U, data, spatialMat, cv=False):
    """
    computing dabc for each subject using individualized parcellations
    Args:
        U: np.ndarray (n_subjects x n_vertices)
            each element is the label for a vertex
        data: np.ndarray (n_subjects x n_conditions x n_vertices)
            if cv=True, data is of shape (n_subjects x n_partitions x n_conditions x n_vertices)
        spatialMat: np.ndarray (n_vertices x n_vertices)
        cv: whether to use cross-validated variance and covariance

    Returns:
        output dictionary
    """

    results = []
    within_corrs = []
    between_corrs = []
    dcbc = []
    output = {}

    if cv:
        n_subjects, n_partitions, n_conditions, n_vertices = data.shape
    else:
        n_subjects, n_conditions, n_vertices = data.shape

    for subjI in np.arange(n_subjects):
        if cv:
            myDCBC = DCBC.compute_DCBC(maxDist=35, binWidth=5, parcellation=U[subjI], func=np.transpose(data[subjI], [0, 2, 1]),
                                       dist=spatialMat, weighting=True, backend='numpy', cv=True)
        else:
            myDCBC = DCBC.compute_DCBC(maxDist=35, binWidth=5, parcellation=U[subjI], func=data[subjI].T,
                                       dist=spatialMat, weighting=True, backend='numpy', cv=False)
        results.append(myDCBC)
        within_corrs.append(myDCBC['corr_within'])
        between_corrs.append(myDCBC['corr_between'])
        dcbc.append(myDCBC['DCBC'])

    within_corrs = np.array(within_corrs)
    between_corrs = np.array(between_corrs)

    output['results'] = results
    output['within_corrs'] = within_corrs
    output['between_corrs'] = between_corrs
    output['dcbc'] = dcbc

    return output
