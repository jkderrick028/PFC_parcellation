import numpy as np
import matplotlib.pyplot as plt
import os, pickle, subprocess
from py_util_dx.py_utils import setProjectPath
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import nibabel as nib
from Functional_Fusion.dataset import decompose_pattern_into_group_indiv_noise
from flat2data import flat2ndarray
import pandas as pd


np.random.seed(seed=0)

def orthogonalize_factors(A, B):
    # orthogonalizing A and B

    B = B - A @ np.linalg.pinv(A) @ B
    return B

n_subjects = 24
n_partitions = 2
n_conditions = 12
n_vertices = 2000

A = np.random.randn(n_subjects, n_partitions, n_conditions, n_vertices)
B = np.random.randn(n_subjects, n_partitions, n_conditions, n_vertices)

# # make sure A and B are orthogonal
for subjectI in np.arange(n_subjects):
    for partI in np.arange(n_partitions):
        for condI in np.arange(n_conditions):
            temp = subjectI + partI + condI
            if np.mod(temp, 2) == 1:
                A[subjectI, partI, condI] = orthogonalize_factors(B[subjectI, partI, condI].reshape(1, n_vertices), A[subjectI, partI, condI].reshape(1, n_vertices))
            else:  
                B[subjectI, partI, condI] = orthogonalize_factors(A[subjectI, partI, condI].reshape(1, n_vertices), B[subjectI, partI, condI].reshape(1, n_vertices))

            dot_product = np.dot(A[subjectI, partI, condI], B[subjectI, partI, condI])
            print(dot_product)

# make sure A and B are orthogonal
# for subjectI in np.arange(n_subjects):
#     for partI in np.arange(n_partitions):        
#         B[subjectI, partI] = orthogonalize_factors(A[subjectI, partI], B[subjectI, partI])

#         dot_product = np.dot(A[subjectI, partI].reshape(-1,), B[subjectI, partI].reshape(-1,))
#         print(dot_product)

data = A + B
variances = decompose_pattern_into_group_indiv_noise(data, criterion='global')
print(variances)

variances_A = decompose_pattern_into_group_indiv_noise(A, criterion='global')
variances_B = decompose_pattern_into_group_indiv_noise(B, criterion='global')

difference = variances - variances_A - variances_B

print(difference)

