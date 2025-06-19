import numpy as np
from scipy.spatial import distance
import DCBC.dcbc as DCBC
import torch as pt


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

        Y_pred = (np.linalg.pinv(U[subjI]) @ V).T
        cos_dist = distance.cosine(Y_pred.flatten(), Y_test[subjI].flatten())
        cosine_distances.append(cos_dist)

    return np.array(cosine_distances)


def compute_dcbc_indiv(U, data, spatialMat):
    """
    computing dabc for each subject using individualized parcellations
    Args:
        U: np.ndarray (n_subjects x n_vertices)
            each element is the label for a vertex
        data: np.ndarray (n_subjects x n_conditions x n_vertices)
        spatialMat: np.ndarray (n_vertices x n_vertices)

    Returns:
        output dictionary
    """

    results = []
    within_corrs = []
    between_corrs = []
    dcbc = []
    output = {}

    n_subjects, n_conditions, n_vertices = data.shape

    for subjI in np.arange(n_subjects):
        myDCBC = DCBC.compute_DCBC(maxDist=35, binWidth=5, parcellation=U[subjI], func=data[subjI].T,
                                   dist=spatialMat, weighting=True, backend='numpy')
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
