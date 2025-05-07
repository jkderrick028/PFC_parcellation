import numpy as np
from scipy.spatial import distance


def prediction_error_cv(U_hat, Y_test):
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

    n_subjects, n_parcels, n_vertices = U_hat.shape
    cosine_distances = []

    for subjI in np.arange(n_subjects):
        train_s = [s for s in np.arange(n_subjects) if s != subjI]
        Y_train = Y_test[train_s]
        Y_train = Y_train.transpose([0, 2, 1])
        U_train = U_hat[train_s]
        U_train = np.divide(U_train, np.sum(U_train, axis=-1, keepdims=True))
        V = np.matmul(U_train, Y_train)
        V = V.mean(axis=0)

        Y_pred = (np.linalg.pinv(U_hat[subjI]) @ V).T
        cos_dist = distance.cosine(Y_pred.flatten(), Y_test[subjI].flatten())
        cosine_distances.append(cos_dist)

    return cosine_distances

