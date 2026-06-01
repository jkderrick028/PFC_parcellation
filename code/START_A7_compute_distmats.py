import os.path, pickle, scipy, sys, subprocess
from py_util_dx.py_utils import setProjectPath
from pathlib import Path
from DCBC.utilities import compute_dist_from_surface


def compute_distmats(dataset_name='HCPur100'):
    """
    computing distance matrix between each pair of vertex 
    """
    ## defining paths
    projectPath, mainResultsPath = setProjectPath()
    # base_dir = '/cifs/diedrichsen/data/FunctionalFusion_new'
    base_dir = '/Volumes/diedrichsen_data$/data/FunctionalFusion_new'

    resultsPath = os.path.join(mainResultsPath, 'START_A7_spatialACF_map_LH', f'{dataset_name}')
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    MAT_output = os.path.join(resultsPath, f'distAvrg_sp_{dataset_name}.mat')

    anat_dir = os.path.join(base_dir, dataset_name, 'derivatives', 'ffimport')

    path = Path(anat_dir)
    subjects = [x.name for x in path.iterdir() if x.is_dir()]

    # subjects = ['sub-101309']
    n_subjects = len(subjects)
   
    dist_matrices = 0

    for subj in subjects:
        white_L = os.path.join(anat_dir, subj, 'anat', f'{subj}_space-32k_hemi-L_white.surf.gii')
        pial_L = os.path.join(anat_dir, subj, 'anat', f'{subj}_space-32k_hemi-L_pial.surf.gii')
        mid_L = os.path.join(resultsPath, f'{subj}_space-32k_hemi-L_mid.surf.gii')

        wb_command = f'wb_command -surface-cortex-layer {white_L} {pial_L} 0.5 {mid_L}'
        subprocess.run(wb_command, shell=True)

        dm = compute_dist_from_surface(mid_L, type='dijkstra', max_dist=50, hems='L', sparse=False)

        dist_matrices += dm  

    dist_matrices = dist_matrices / n_subjects

    scipy.io.savemat(MAT_output, {'avrgDs': scipy.sparse.csr_matrix(dist_matrices)})


if __name__=='__main__':
    try:
        dataset_name = sys.argv[1]
    except:
        dataset_name = 'HCPur100'

    compute_distmats(dataset_name=dataset_name)
