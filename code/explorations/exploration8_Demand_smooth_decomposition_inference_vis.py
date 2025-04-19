import os.path, pickle
import numpy as np
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath, sqmat2vec

projectPath, mainResultsPath = setProjectPath()
# base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
# base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''))
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

PKL_output = os.path.join(mainResultsPath, 'exploration8_Demand_decomposition_inference_work', 'output.pkl')
with open(PKL_output, 'rb') as pf:
    output = pickle.load(pf)

PKL_vis_output = os.path.join(resultsPath, 'vis_output.pkl')
vis_output = {}

smooth_kernels = list(output.keys())
regions = list(output['orig'].keys())

v_s_over_gs = {}
v_g_plus_v_s_over_gse = {}


for region in regions:
    v_s_over_gs[region] = []
    v_g_plus_v_s_over_gse[region] = []

    for smooth_kernel in smooth_kernels:
        v_s_over_gs[region].append(np.divide(output[smooth_kernel][region][1:, 1], (output[smooth_kernel][region][1:, 0] + output[smooth_kernel][region][1:, 1])))
        v_g_plus_v_s_over_gse[region].append(np.divide((output[smooth_kernel][region][1:, 0] + output[smooth_kernel][region][1:, 1]), np.sum(output[smooth_kernel][region][1:, :], axis=1)))

    v_s_over_gs[region] = np.array(v_s_over_gs[region])
    v_g_plus_v_s_over_gse[region] = np.array(v_g_plus_v_s_over_gse[region])

n_samples = v_s_over_gs['whole_cortex'].shape[1]

vis_output['v_s_over_gs'] = v_s_over_gs
vis_output['v_g_plus_v_s_over_gse'] = v_g_plus_v_s_over_gse
vis_output['n_samples'] = n_samples

v_s_over_gs_mean = {}
v_s_over_gs_ste = {}
v_g_plus_v_s_over_gse_mean = {}
v_g_plus_v_s_over_gse_ste = {}

for region in regions:
    v_s_over_gs_mean[region] = v_s_over_gs[region].mean(axis=1)
    v_s_over_gs_ste[region] = np.divide(v_s_over_gs[region].std(axis=1), np.sqrt(n_samples))
    v_g_plus_v_s_over_gse_mean[region] = v_g_plus_v_s_over_gse[region].mean(axis=1)
    v_g_plus_v_s_over_gse_ste[region] = np.divide(v_g_plus_v_s_over_gse[region].std(axis=1), np.sqrt(n_samples))

plt.figure()

color_cortex = (0, 56/255, 118/255)
color_DLPFC = (1, 150/255, 0)
color_somatosensory = (171/255, 219/255, 227/255)
color_parietal = (51/255, 153/255, 102/255)
color_visual = (135/255, 62/255, 35/255)

plt.errorbar(0, v_s_over_gs_mean['whole_cortex'][0], v_s_over_gs_ste['whole_cortex'][0], color=color_cortex, marker='o')
plt.errorbar(0, v_s_over_gs_mean['DLPFC'][0], v_s_over_gs_ste['DLPFC'][0], color=color_DLPFC, marker='o')
plt.errorbar(0, v_s_over_gs_mean['somatosensory'][0], v_s_over_gs_ste['somatosensory'][0], color=color_somatosensory, marker='o')
plt.errorbar(0, v_s_over_gs_mean['parietal'][0], v_s_over_gs_ste['parietal'][0], color=color_parietal, marker='o')
plt.errorbar(0, v_s_over_gs_mean['visual'][0], v_s_over_gs_ste['visual'][0], color=color_visual, marker='o')

plt.errorbar(np.arange(1, len(smooth_kernels)), v_s_over_gs_mean['whole_cortex'][1:], v_s_over_gs_ste['whole_cortex'][1:], linestyle='-', color=color_cortex, label='whole cortex')
plt.errorbar(np.arange(1, len(smooth_kernels)), v_s_over_gs_mean['DLPFC'][1:], v_s_over_gs_ste['DLPFC'][1:], linestyle='-', color=color_DLPFC, label='DLPFC')
plt.errorbar(np.arange(1, len(smooth_kernels)), v_s_over_gs_mean['somatosensory'][1:], v_s_over_gs_ste['somatosensory'][1:], linestyle='-', color=color_somatosensory, label='somatosensory')
plt.errorbar(np.arange(1, len(smooth_kernels)), v_s_over_gs_mean['parietal'][1:], v_s_over_gs_ste['parietal'][1:], linestyle='-', color=color_parietal, label='parietal')
plt.errorbar(np.arange(1, len(smooth_kernels)), v_s_over_gs_mean['visual'][1:], v_s_over_gs_ste['visual'][1:], linestyle='-', color=color_visual, label='visual')

plt.hlines(0, xmin=0, xmax=len(smooth_kernels)-1, linestyles='--', colors=(150/255, 150/255, 150/255))
plt.hlines(1, xmin=0, xmax=len(smooth_kernels)-1, linestyles='--', colors=(150/255, 150/255, 150/255))
plt.text(0, 0.05, 'group')
plt.text(0, 0.95, 'individual')

xlabels = smooth_kernels

plt.xticks(np.arange(len(smooth_kernels)), xlabels)
plt.ylabel('v_s/v_s+v_g')
plt.title('v_s over (v_s + v_g)')

plt.legend(frameon=False, loc='right')
plt.ylim([-0.1, 1.1])

plt.savefig(os.path.join(resultsPath, f'v_s_over_v_s_plus_v_g.png'), dpi=500)

plt.figure()

plt.errorbar(0, v_g_plus_v_s_over_gse_mean['whole_cortex'][0], v_g_plus_v_s_over_gse_ste['whole_cortex'][0], color=color_cortex, marker='o')
plt.errorbar(0, v_g_plus_v_s_over_gse_mean['DLPFC'][0], v_g_plus_v_s_over_gse_ste['DLPFC'][0], color=color_DLPFC, marker='o')
plt.errorbar(0, v_g_plus_v_s_over_gse_mean['somatosensory'][0], v_g_plus_v_s_over_gse_ste['somatosensory'][0], color=color_somatosensory, marker='o')
plt.errorbar(0, v_g_plus_v_s_over_gse_mean['parietal'][0], v_g_plus_v_s_over_gse_ste['parietal'][0], color=color_parietal, marker='o')
plt.errorbar(0, v_g_plus_v_s_over_gse_mean['visual'][0], v_g_plus_v_s_over_gse_ste['visual'][0], color=color_visual, marker='o')

plt.errorbar(np.arange(1, len(smooth_kernels)), v_g_plus_v_s_over_gse_mean['whole_cortex'][1:], v_g_plus_v_s_over_gse_ste['whole_cortex'][1:], linestyle='-', color=color_cortex, label='whole cortex')
plt.errorbar(np.arange(1, len(smooth_kernels)), v_g_plus_v_s_over_gse_mean['DLPFC'][1:], v_g_plus_v_s_over_gse_ste['DLPFC'][1:], linestyle='-', color=color_DLPFC, label='DLPFC')
plt.errorbar(np.arange(1, len(smooth_kernels)), v_g_plus_v_s_over_gse_mean['somatosensory'][1:], v_g_plus_v_s_over_gse_ste['somatosensory'][1:], linestyle='-', color=color_somatosensory, label='somatosensory')
plt.errorbar(np.arange(1, len(smooth_kernels)), v_g_plus_v_s_over_gse_mean['parietal'][1:], v_g_plus_v_s_over_gse_ste['parietal'][1:], linestyle='-', color=color_parietal, label='parietal')
plt.errorbar(np.arange(1, len(smooth_kernels)), v_g_plus_v_s_over_gse_mean['visual'][1:], v_g_plus_v_s_over_gse_ste['visual'][1:], linestyle='-', color=color_visual, label='visual')

plt.xticks(np.arange(len(smooth_kernels)), xlabels)
plt.ylabel('v_s+v_g')
plt.title('v_s + v_g')

plt.legend(frameon=False, loc='upper right')
plt.ylim([0, 1])

plt.savefig(os.path.join(resultsPath, f'v_s_plus_v_g.png'), dpi=500)

with open(PKL_vis_output, 'wb') as pk:
    pickle.dump(vis_output, pk)

plt.show()
