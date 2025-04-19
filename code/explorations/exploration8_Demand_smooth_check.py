import os.path, pickle
import numpy as np
from py_util_dx.py_utils import setProjectPath
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from Functional_Fusion.dataset import decompose_pattern_into_group_indiv_noise
from flat2data import flat2ndarray


projectPath, mainResultsPath = setProjectPath()

# smoothing_results_path = os.path.join(mainResultsPath, 'exploration8_Demand_smooth_work')
# smoothing_kernels = [10, 8, 6, 4, 2, 'residuals']      # mm, fwhm

# smoothing_results_path = os.path.join(mainResultsPath, 'exploration8_Demand_smooth_toy_subtraction')
smoothing_results_path = os.path.join(mainResultsPath, 'exploration8_Demand_smooth_reg')
smoothing_kernels = [10, 8, 6, 4, 2, 'residuals']      # mm, fwhm

# load original unsmoothed matrix
# PKL_smoothed = os.path.join(smoothing_results_path, 'smoothed_orig_mm.pkl')
PKL_smoothed = os.path.join(smoothing_results_path, 'regressed_orig_mm.pkl')

with open(PKL_smoothed, 'rb') as pf:
    output = pickle.load(pf)
    data_unsmoothed = output['X_individuals']

data_unsmoothed = data_unsmoothed[[0, 1]]
data_unsmoothed = data_unsmoothed[:, [0, 1, 12, 13], :]

smoothed_data = []

for smoothing_kernel in smoothing_kernels:
    # PKL_smoothed = os.path.join(smoothing_results_path, f'smoothed_{smoothing_kernel}_mm.pkl')
    PKL_smoothed = os.path.join(smoothing_results_path, f'regressed_{smoothing_kernel}_mm.pkl')

    with open(PKL_smoothed, 'rb') as pf:
        data_smoothed = pickle.load(pf)

    smoothed_data.append(data_smoothed)

smoothed_data = np.array(smoothed_data)

# check if the sum of frequency bands is the same as original unmoothed data
sum_smoothed_data = np.sum(smoothed_data, axis=0)

difference = data_unsmoothed - sum_smoothed_data

print(f'max = {np.max(difference)}')
print(f'min = {np.min(difference)}')

# check if smoothing kernels are mean-preserving
leftovers = []
for smoothing_kernel in [10, 8, 6, 4, 2] :
    PKL_smoothed = os.path.join(smoothing_results_path, f'leftover_{smoothing_kernel}_mm.pkl')
    with open(PKL_smoothed, 'rb') as pf:
        data_smoothed = pickle.load(pf)
    leftovers.append(data_smoothed)

leftovers = np.array(leftovers)

n_subjects, n_conds, n_vertices = data_unsmoothed.shape
n_smoothing_kernels = len(smoothing_kernels)

mean_leftovers = np.zeros((n_smoothing_kernels-1, n_subjects, n_conds))
mean_smoothed = np.zeros((n_smoothing_kernels, n_subjects, n_conds))
mean_orig = np.mean(data_unsmoothed, axis=2)

for i in np.arange(n_smoothing_kernels):
    mean_smoothed[i] = np.mean(smoothed_data[i], axis=2)
    if i < (n_smoothing_kernels-1):        
        mean_leftovers[i] = np.mean(leftovers[i], axis=2)

mean_differences = []
# orig, 10 mm
mean_differences.append(mean_orig - mean_smoothed[0])
# 10 mm res, 8 mm
mean_differences.append(mean_leftovers[0] - mean_smoothed[1])
# 8 mm res, 6 mm
mean_differences.append(mean_leftovers[1] - mean_smoothed[2])
# 6 mm res, 4 mm
mean_differences.append(mean_leftovers[2] - mean_smoothed[3])
# 4 mm res, 2 mm
mean_differences.append(mean_leftovers[3] - mean_smoothed[4])

print(f'max = {np.max(mean_differences)}')
print(f'min = {np.min(mean_differences)}')

# check if variances for each frequency band sum up to original (vg, vs, ve)
variances = []
criterion = 'global'
part_vec = [1, 1, 2, 2]
cond_vec = [1, 2, 1, 2]

#  original
data = flat2ndarray(data_unsmoothed, part_vec, cond_vec)
variances.append(decompose_pattern_into_group_indiv_noise(data, criterion=criterion).flatten())

# for each frequency band
for i in np.arange(len(smoothing_kernels)):
    data = flat2ndarray(smoothed_data[i], part_vec, cond_vec)
    variances.append(decompose_pattern_into_group_indiv_noise(data, criterion=criterion).flatten())

variances = np.array(variances)

variances_sum_frequency_bands = np.sum(variances[1:], axis=0)
difference = variances[0] - variances_sum_frequency_bands
print(difference)

# original data as y, 10mm smoothed data as predictor, building a regression model
y = data_unsmoothed[0, 0].reshape(-1, 1)
X = smoothed_data[0, 0, 0].reshape(-1, 1)

reg = LinearRegression(fit_intercept=False).fit(X,y)

x = np.linspace(-2, 6, 200)
y_pred = reg.coef_ * x + reg.intercept_

fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(X, y)
plt.xlim([-3.5, 6.5])
plt.ylim([-3.5, 6.5])
ax.set_aspect('equal', adjustable='box')
plt.plot(x, y_pred.flatten(), color='red')
plt.ylabel('orig')
plt.xlabel('10 mm smoothed')
plt.show()
print((reg.coef_, reg.intercept_))
print(reg.score(X, y))



