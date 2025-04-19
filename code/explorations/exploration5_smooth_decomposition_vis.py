import os.path, pickle, subprocess
import numpy as np
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath, sqmat2vec

projectPath, mainResultsPath = setProjectPath()
# base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion'
# base_dir = '/cifs/diedrichsen/data/FunctionalFusion'
base_dir = os.path.join(projectPath, 'data', 'FunctionalFusion')
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('vis.py', 'work'), 'all_subjects')
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

PKL_output = os.path.join(resultsPath, 'output.pkl')
with open(PKL_output, 'rb') as pf:
    output_smooth_decomposition = pickle.load(pf)

PKL_output_B4 = os.path.join(resultsPath, 'output_B4.pkl')
with open(PKL_output_B4, 'rb') as pf:
    output_B4 = pickle.load(pf)

smooth_kernels = [10, 8, 6, 4, 2]

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
        v_g_PFC.append(output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, 0] / np.sum(output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, 0:2]))
        v_s_PFC.append(output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, 1] / np.sum(output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, 0:2]))

        v_g_cortex.append(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, 0] / np.sum(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, 0:2]))
        v_s_cortex.append(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, 1] / np.sum(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, 0:2]))

plt.figure()

v_g_PFC = np.array(v_g_PFC)
v_s_PFC = np.array(v_s_PFC)
v_g_cortex = np.array(v_g_cortex)
v_s_cortex = np.array(v_s_cortex)

v_s_over_s_g_PFC = np.divide(v_s_PFC, (v_s_PFC + v_g_PFC))
v_s_over_s_g_cortex = np.divide(v_s_cortex, (v_s_cortex + v_g_cortex))

color_cortex = (0, 56/255, 118/255)
color_PFC = (1, 150/255, 0)

plt.scatter(0, v_s_over_s_g_cortex[0], color=color_cortex)
plt.scatter(0, v_s_over_s_g_PFC[0], color=color_PFC)

plt.plot(1+np.arange(len(smooth_kernels)), v_s_over_s_g_cortex[1:], linestyle='-', color=color_cortex, label='whole cortex')
plt.plot(1+np.arange(len(smooth_kernels)), v_s_over_s_g_PFC[1:], linestyle='-', color=color_PFC, label='PFC')

plt.hlines(0, xmin=0, xmax=5, linestyles='--', colors=(150/255, 150/255, 150/255))
plt.hlines(1, xmin=0, xmax=5, linestyles='--', colors=(150/255, 150/255, 150/255))
plt.text(0, 0.05, 'group')
plt.text(0, 0.95, 'individual')

xlabels = ['orig']
for kernel in smooth_kernels:
    xlabels.append(f'{kernel}_mm')

plt.xticks(np.arange(len(v_g_PFC)), xlabels)
plt.ylabel('v_s/v_s+v_g')
plt.title(f'v_s {abs_or_prop} 24 subjects')

plt.legend(frameon=False, loc='right')
plt.ylim([-0.1, 1.1])

plt.savefig(os.path.join(resultsPath, f'v_g_{abs_or_prop}.png'), dpi=500)


plt.figure()

vs_plus_vg_cortex = [(output_B4['whole_cortex']['global'][0, 0] + output_B4['whole_cortex']['global'][0, 1]) / np.sum(output_B4['whole_cortex']['global'][0, :])]
vs_plus_vg_PFC = [(output_B4['PFC']['global'][0, 0] + output_B4['PFC']['global'][0, 1]) / np.sum(output_B4['PFC']['global'][0, :])]
for kernel in smooth_kernels:
    vs_plus_vg_cortex.append((output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, 0] + output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, 1]) / np.sum(output_smooth_decomposition[f'{kernel}_mm']['whole_cortex'][0, :]))
    vs_plus_vg_PFC.append((output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, 0] + output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, 1]) / np.sum(output_smooth_decomposition[f'{kernel}_mm']['PFC'][0, :]))

plt.scatter(0, vs_plus_vg_cortex[0], color=color_cortex)
plt.scatter(0, vs_plus_vg_PFC[0], color=color_PFC)

plt.plot(1+np.arange(len(smooth_kernels)), vs_plus_vg_cortex[1:], linestyle='-', color=(0, 56/255, 118/255), label='whole cortex')
plt.plot(1+np.arange(len(smooth_kernels)), vs_plus_vg_PFC[1:], linestyle='-', color=(1, 150/255, 0), label='PFC')

plt.xticks(np.arange(len(v_g_PFC)), xlabels)
plt.ylabel('v_s+v_g')
plt.title(f'v_s + v_g 24 subjects')

plt.legend(frameon=False, loc='upper right')
plt.ylim([0, 1])

plt.savefig(os.path.join(resultsPath, f'v_s_plus_v_g.png'), dpi=500)

plt.show()
