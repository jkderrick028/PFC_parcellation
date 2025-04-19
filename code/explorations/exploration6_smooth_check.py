import os.path, pickle
import numpy as np
from py_util_dx.py_utils import setProjectPath


projectPath, mainResultsPath = setProjectPath()
resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''))
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

smoothing_results_path = os.path.join(mainResultsPath, 'exploration6_smooth_work')

smoothing_kernels = [10, 8, 6, 4, 2, 'residuals']      # mm, fwhm

# load original unsmoothed matrix
PKL_smoothed = os.path.join(smoothing_results_path, 'smoothed_orig_mm.pkl')

with open(PKL_smoothed, 'rb') as pf:
    output = pickle.load(pf)
    data_unsmoothed = output['X_individuals']

smoothed_data = []

for smoothing_kernel in smoothing_kernels:
    PKL_smoothed = os.path.join(smoothing_results_path, f'smoothed_{smoothing_kernel}_mm.pkl')

    with open(PKL_smoothed, 'rb') as pf:
        data_smoothed = pickle.load(pf)

    smoothed_data.append(data_smoothed)

smoothed_data = np.array(smoothed_data)

sum_smoothed_data = np.sum(smoothed_data, axis=0)

difference = data_unsmoothed - sum_smoothed_data

print(f'max = {np.max(difference)}')
print(f'min = {np.min(difference)}')

PKL_output = os.path.join(resultsPath, 'smooth_sum_check.pkl')
with open(PKL_output, 'wb') as pf:
    pickle.dump(difference, pf)

print('done')
