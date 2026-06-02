import os.path, pickle, scipy, sys
import nibabel as nib
import matplotlib.pyplot as plt
from py_util_dx.py_utils import setProjectPath
from py_util_dx.data_utils import get_roi_pacels, get_glasser_labels, get_roi_vtx_from_fs32k
from scipy.stats import ttest_ind, ttest_1samp
from evaluations import *
from Functional_Fusion.reliability import flat2ndarray


def compute_spatial_ACF(dataset_name='MDTB'):
    """
    computing cross-validated spatial ACF, for the entire left hemisphere
    Args:
        ROI: str

    """
    ## defining paths
    projectPath, mainResultsPath = setProjectPath()

    # dataset_name = 'MDTB' # or Demand

    surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

    resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}')
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)

    output = dict()
    PKL_output = os.path.join(resultsPath, f'output_LH_cv.pkl')

    ## loading individualized parcellation and glasser group parcellation
    included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k('whole_cortex')

    ## loading MDTB data
    if dataset_name == 'MDTB':
        PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_CondHalf_ses-s2.pkl')
    else:
        PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_CondHalf_all.pkl')
        
    with open(PKL_data, 'rb') as pf:
        original_data = pickle.load(pf)
        X_individuals = original_data['X_individuals']
        info_individuals = original_data['info_individuals']
        dataset_obj_individuals = original_data['dataset_obj_individuals']

    # fill nans with 0
    X_individuals[np.isnan(X_individuals)] = 0
    n_subjects = X_individuals.shape[0]

    part_vec = list(info_individuals['half'])
    # cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

    if dataset_name == 'Nishimoto':
        cond_vec = list(info_individuals['cond_num'])
    else:
        cond_vec = list(info_individuals[dataset_obj_individuals.cond_ind])

    data = flat2ndarray(X_individuals[:, :, included_vtx_inds_L], part_vec, cond_vec)

    if dataset_name == 'MDTB':
        MAT_dist = os.path.join(projectPath, 'code', 'DCBC', 'distanceMatrix', 'distAvrg_sp.mat')
    else:
        MAT_dist = os.path.join(resultsPath, f'distAvrg_sp_{dataset_name}.mat')

    spatialMat = scipy.io.loadmat(MAT_dist)['avrgDs'].toarray()

    glasser_L = os.path.join(surface_helpers_dir, 'glasser.L.label.gii')

    # extract the first data array as the parcels
    # make sure that the input parcels are of shape (N,)
    gii_file = nib.load(glasser_L)

    parcels_inds = gii_file.darrays[0].data

    # get all the vertices that are in the ROI list
    vertex_label_ROI, vertex_ind_ROI = [], []
    for i, label in enumerate(parcels_inds):
        if label not in [0]:
            vertex_label_ROI.append(label)
            vertex_ind_ROI.append(i)

    vertex_label_ROI = np.array(vertex_label_ROI)
    vertex_ind_ROI = np.array(vertex_ind_ROI)

    spatialMat = spatialMat[vertex_ind_ROI, :]
    spatialMat = spatialMat[:, vertex_ind_ROI]

    FWHMs_across_subjects = []
    for subjI in np.arange(n_subjects):
        D = spatial_ACF_per_voxel_cv(maxDist=35, binWidth=5, func=np.transpose(data[subjI], [0, 2, 1]), dist=spatialMat)
        FWHMs_across_subjects.append(D['FWHMs'])

    FWHMs_across_subjects = np.array(FWHMs_across_subjects)
    distances = D['dists']

    output['FWHMs_across_subjects'] = FWHMs_across_subjects
    output['distances'] = distances

    with open(PKL_output, 'wb') as pf:
        pickle.dump(output, pf)


if __name__=='__main__':
    try:
        dataset_name = sys.argv[1]
    except:
        # dataset_name = 'MDTB'
        # dataset_name = 'HCPur100'
        # dataset_name = 'Language'
        # dataset_name = 'Nishimoto'
        dataset_name = 'IBC'

    compute_spatial_ACF(dataset_name=dataset_name)
