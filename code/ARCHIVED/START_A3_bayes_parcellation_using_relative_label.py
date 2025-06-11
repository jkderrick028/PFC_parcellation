import numpy as np
import nitools as nt
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.util as ut
from py_util_dx.py_utils import setProjectPath
import os, pickle
from nitools.cifti import surf_from_cifti
import SUITPy.flatmap as flatmap
import torch
from py_util_dx.data_utils import get_roi_pacels, get_roi_vtx_from_fs32k


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

strength = 7.0

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), f'{dataset_name}_{strength}')
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

# appendix = 'whole_cortex'   # or PFC_masked
appendix = 'PFC_masked'
# appendix = 'somatosensory_masked'
# appendix = 'visual_masked'
# appendix = 'parietal_masked'

PKL_output = os.path.join(resultsPath, f'output_{appendix}.pkl')
output = {}

## loading MDTB data
PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_All_ses-s1.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    data = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

cond_vec = np.array(list(info_individuals[dataset_obj_individuals.cond_ind]))
# part_vec = np.array(list(info_individuals[dataset_obj_individuals.part_ind]))
part_vec = np.ones(len(cond_vec)).astype(int)

## loading group atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

# Sample the probabilistic atlas at the specific atlas grayordinates
atlas_fname = [os.path.join(surface_helpers_dir, 'glasser.L.label.gii'), os.path.join(surface_helpers_dir, 'glasser.R.label.gii')]
U = atlas.read_data(atlas_fname)

U_shape_orig = U.shape

## dealing with PFC mask
if appendix == 'whole_cortex':
    included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k('whole_cortex')
else:
    included_vtx_inds_LR, included_vtx_inds_L, included_vtx_inds_R, excluded_vtx_inds_LR = get_roi_vtx_from_fs32k(appendix.rstrip('_masked'))

data = data[:, :, included_vtx_inds_LR]
U_roi = U[included_vtx_inds_LR]


## converting the hard parcellation into a probabilistic one
# U = IndividualParcellation.utils.convert_hard_to_prob(U, strength=7.0)
labels_in_glasser, U_roi = np.unique(U_roi, return_inverse=True)    # labels_in_glasser is a list of labels of parcels of interest in glasser parcellation, starting from 1
parcel_names = get_roi_pacels('whole_cortex')
parcel_names_in_glasser = [parcel_names[k-1] for k in labels_in_glasser]
K = np.unique(U_roi).size

logpi = ar.expand_mn_1d(U_roi, K) * strength
U_roi = torch.softmax(logpi, dim=0)

# Build the arrangement model - the parameters are the log-probabilities of the atlas
# ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas)
ar_model = ar.build_arrangement_model(U_roi, prior_type='logpi', atlas=atlas)

# fit the emission model to the data
# K is the number of parcels
K = ar_model.K
# Make a design matrix
X= ut.indicator(cond_vec)
# Build an emission model
# em_model = em.MixVMF(K=K,P=atlas.P, X=X,part_vec=part_vec)
em_model = em.MixVMF(K=K, P=U_roi.shape[1], X=X, part_vec=part_vec)

# Build the full model: The emission models are passed as a list, as usually we have multiple data sets
M = fm.FullMultiModel(ar_model, [em_model])
# Attach the data to the model - this is done for speed
# The data is passed as a list with on element per data set
M.initialize([data])

# Now we can run the EM algorithm
M, ll, _, U_indiv = M.fit_em(iter=1000, tol=0.01, fit_arrangement=False,fit_emission=True,first_evidence=False)

# get the data only parcellation
emloglik  = M.emissions[0].Estep()
Uhat_data = torch.softmax(emloglik, dim=1).numpy()

# printing kappa
print(f'kappa: {M.emissions[0].kappa}')

## restoring U to the original shape (containing all vertices in the cortex)
U_restore = np.zeros((U_roi.shape[0], U_shape_orig[0]))
U_restore[:, included_vtx_inds_LR] = U_roi
U_roi = U_restore.copy()

U_indiv_restore = np.zeros((U_indiv.shape[0], U_indiv.shape[1], U_shape_orig[0]))
U_indiv_restore[:, :, included_vtx_inds_LR] = U_indiv
U_indiv = U_indiv_restore.copy()

Uhat_data_restore = np.zeros((Uhat_data.shape[0], Uhat_data.shape[1], U_shape_orig[0]))
Uhat_data_restore[:, :, included_vtx_inds_LR] = Uhat_data
Uhat_data = Uhat_data_restore.copy()

# saving U and U_indiv
output['U'] = U
output['U_roi'] = U_roi
output['U_indiv'] = U_indiv
output['Uhat_data'] = Uhat_data
# output['M'] = M
output['ll'] = ll
output['labels_in_glasser'] = labels_in_glasser
output['parcel_names_in_glasser'] = parcel_names_in_glasser
output['V'] = M.emissions[0].V.numpy()
# output['theta'] = theta

with open(PKL_output, 'wb') as pf:
    pickle.dump(output, pf)

# # loading saved U and U_indiv
# with open(PKL_output, 'rb') as pf:
#     output = pickle.load(pf)
#
# # U = output['U']
# U_indiv = output['U_indiv']
# ll = output['ll']

## Load colormap and labels
lid,cmap,names = nt.read_lut(os.path.join(surface_helpers_dir, 'atl-glasser.lut'))
# modify these color settings when putting on a PFC mask
if appendix != 'whole_cortex':
    parcels = get_roi_pacels(appendix.rstrip('_masked'))
    keep_inds = []
    for i in np.arange(len(names)):
        if names[i].split('_')[1] in parcels:
            keep_inds.append(i)
    lid = lid[keep_inds]
    cmap = cmap[keep_inds]
    names = list(map(names.__getitem__, keep_inds))

flat_surf_L = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'fs_LR.32k.R.flat.surf.gii')
border_LR = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.border')

def plot_probseg(surf_data, cmap, hemi):
    label = np.argmax(surf_data, axis=0) + 1
    label[excluded_vtx_inds_LR] = 181

    [label_L, label_R] = surf_from_cifti(atlas.data_to_cifti(label.reshape(1, -1)))

    if hemi == 'L':
        # left cortex
        keep_inds = np.arange(int(cmap.shape[0]/2))
        cmap = np.vstack([np.ones((1, 3)), cmap[keep_inds, :], np.ones((1, 3))])

        flatmap.plot(label_L.reshape(-1, ),
                     surf=flat_surf_L,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     label_names=list(map(names.__getitem__, keep_inds)),
                     new_figure=False,
                     frame=None,
                     render='matplotlib',
                     cmap=cmap,
                     borders=border_LR,
                     overlay_type='label',
                     bordersize=3,
                     undermap='gray',
                     underscale=[-1, 0.5]
        )

    else:
        # right cortex
        keep_inds = np.arange(int(cmap.shape[0]/2), cmap.shape[0])
        cmap = np.vstack([np.ones((1, 3)), cmap[keep_inds, :], np.ones((1, 3))])

        flatmap.plot(label_R.reshape(-1, ),
                     surf=flat_surf_R,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     label_names=list(map(names.__getitem__, keep_inds)),
                     new_figure=False,
                     frame=None,
                     render='matplotlib',
                     cmap=cmap,
                     borders=border_LR,
                     overlay_type='label',
                     bordersize=3,
                     undermap='gray',
                     underscale=[-1, 0.5]
        )


# Make a nifti image of the first subject
if torch.is_tensor(U_roi):
    surf_data = U_roi.detach().numpy()
else:
    surf_data = U_roi.copy()

# plot the group probabilistic atlas
plt.figure()
plt.subplot(1, 2, 1)
plot_probseg(surf_data, cmap, 'L')
plt.subplot(1, 2, 2)
plot_probseg(surf_data, cmap, 'R')
plt.suptitle('group')

# plot 3 individual subjects (data only parcellation)
for i,s in enumerate([6,9,12]):
    if torch.is_tensor(Uhat_data):
        surf_data = Uhat_data[s].detach().numpy()
    else:
        surf_data = Uhat_data[s]

    plt.figure()
    plt.subplot(1, 2, 1)
    plot_probseg(surf_data, cmap, 'L')
    plt.subplot(1, 2, 2)
    plot_probseg(surf_data, cmap, 'R')
    plt.suptitle(f'subject {s}')


JPG_fig = os.path.join(resultsPath, f'example_parcellations_{appendix}.jpg')
plt.savefig(JPG_fig, format='jpg', dpi=400)

# inspect model training
plt.figure(figsize=(5,5))
plt.plot(ll)

JPG_fig = os.path.join(resultsPath, f'll_training_{appendix}.jpg')
plt.savefig(JPG_fig, format='jpg', dpi=400)

pass
