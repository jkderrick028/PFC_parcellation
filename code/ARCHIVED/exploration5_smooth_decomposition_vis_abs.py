import os.path, pickle, subprocess
import numpy as np
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
from sklearn.manifold import MDS
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from py_util_dx.py_utils import setProjectPath, sqmat2vec
from sklearn.linear_model import LinearRegression
from Functional_Fusion.dataset import decompose_pattern_into_group_indiv_noise
from flat2data import flat2ndarray
import SUITPy.flatmap as flatmap
from nitools.cifti import surf_from_cifti


projectPath, mainResultsPath = setProjectPath()
# base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
# base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('vis_abs.py', 'work'), 'all_subjects')
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

PKL_output = os.path.join(resultsPath, 'output.pkl')
with open(PKL_output, 'rb') as pf:
    output_smooth_decomposition = pickle.load(pf)

PKL_output_B4 = os.path.join(resultsPath, 'output_B4.pkl')
with open(PKL_output_B4, 'rb') as pf:
    output_B4 = pickle.load(pf)

smooth_kernels = [10, 8, 6, 4, 2]
# smooth_kernels = [20, 15, 10]
v_g_PFC = []
v_s_PFC = []
v_g_cortex = []
v_s_cortex = []

abs_or_prop = 'prop'    # 'abs' or 'prop'

if abs_or_prop == 'abs':
    v_g_PFC.append(output_B4['PFC']['global'][0, 0])
    v_s_PFC.append(output_B4['PFC']['global'][0, 1])
    v_g_cortex.append(output_B4['whole_cortex']['global'][0, 0])
    v_s_cortex.append(output_B4['whole_cortex']['global'][0, 1])
else:
    v_g_PFC.append(output_B4['PFC']['global'][0, 0] / np.sum(output_B4['PFC']['global'][0, 0:2]))
    v_s_PFC.append(output_B4['PFC']['global'][0, 1] / np.sum(output_B4['PFC']['global'][0, 0:2]))
    v_g_cortex.append(output_B4['whole_cortex']['global'][0, 0] / np.sum(output_B4['whole_cortex']['global'][0, 0:2]))
    v_s_cortex.append(output_B4['whole_cortex']['global'][0, 1] / np.sum(output_B4['whole_cortex']['global'][0, 0:2]))

for kernel in smooth_kernels:
    if abs_or_prop == 'abs':
        v_g_PFC.append(output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, 0])
        v_s_PFC.append(output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, 1])

        v_g_cortex.append(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, 0])
        v_s_cortex.append(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, 1])
    else:
        v_g_PFC.append(output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, 0] / np.sum(output_smooth_decomposition[f'{kernel}_mm']['PFC']))
        v_s_PFC.append(output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, 1] / np.sum(output_smooth_decomposition[f'{kernel}_mm']['PFC']))

        v_g_cortex.append(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, 0] / np.sum(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex']))
        v_s_cortex.append(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, 1] / np.sum(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex']))

plt.figure()

plt.plot(np.arange(len(v_g_cortex)), v_g_cortex, 'g-')
plt.plot(np.arange(len(v_s_cortex)), v_s_cortex, 'g--')
plt.plot(np.arange(len(v_g_PFC)), v_g_PFC, linestyle='-', color=(0, 32/255, 91/255))
plt.plot(np.arange(len(v_s_PFC)), v_s_PFC, linestyle='--', color=(0, 32/255, 91/255))
plt.legend(['cortex_g', 'cortex_s', 'PFC_g', 'PFC_s'])
xlabels = ['orig']
for kernel in smooth_kernels:
    xlabels.append(f'{kernel}_mm')

plt.xticks(np.arange(len(v_g_PFC)), xlabels)
plt.ylabel('v_g')
plt.title(f'v_g {abs_or_prop} 24 subjects')
# plt.title(f'v_g {abs_or_prop} 5 subjects')

plt.savefig(os.path.join(resultsPath, f'v_g_{abs_or_prop}.svg'), dpi=500)
plt.show()
