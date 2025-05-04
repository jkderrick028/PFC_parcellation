import numpy as np
import torch as pt
import nibabel as nb
import nitools as nt
import pandas as pd
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.util as ut
import SUITPy as suit
from py_util_dx.py_utils import setProjectPath
import os, pickle
import IndividualParcellation
from nitools.cifti import surf_from_cifti
import SUITPy.flatmap as flatmap


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

appendix = 'whole_cortex'   # or PFC masked

PKL_output = os.path.join(resultsPath, f'output_{appendix}.pkl')
output = {}

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

# Sample the probabilistic atlas at the specific atlas grayordinates
# atlas_fname = os.path.join(surface_helpers_dir, 'atl-NettekovenSym32_space-MNI152NLin2009cSymC_probseg.nii.gz')
atlas_fname = [os.path.join(surface_helpers_dir, 'glasser.L.label.gii'), os.path.join(surface_helpers_dir, 'glasser.R.label.gii')]
U = atlas.read_data(atlas_fname)
U = U.T

## converting the hard parcellation into a probabilistic one
# U = IndividualParcellation.utils.convert_hard_to_prob(U, strength=7.0)

_, U = np.unique(U, return_inverse=True)
K = np.unique(U).size

logpi = ar.expand_mn_1d(U, K)
# Set parcel 0 to unassigned
logpi = logpi[1:, :] if np.any(np.unique(U) == 0) else logpi
U = logpi

## dealing with PFC mask



# Build the arrangement model - the parameters are the log-probabilities of the atlas
# ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas)
ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas)

# loading MDTB data
PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_All_ses-s1.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    data = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

cond_vec = np.array(list(info_individuals[dataset_obj_individuals.cond_ind]))
part_vec = np.array(list(info_individuals[dataset_obj_individuals.part_ind]))

# fit the emission model to the data
# K is the number of parcels
K = ar_model.K
# Make a design matrix
X= ut.indicator(cond_vec)
# Build an emission model
em_model = em.MixVMF(K=K,P=atlas.P, X=X,part_vec=part_vec)
# Build the full model: The emission models are passed as a list, as usually we have multiple data sets
M = fm.FullMultiModel(ar_model, [em_model])
# Attach the data to the model - this is done for speed
# The data is passed as a list with on element per data set
M.initialize([data])

# Now we can run the EM algorithm
# M, _, _, _ = M.fit_em(iter=200, tol=0.01, fit_arrangement=False,fit_emission=True,first_evidence=False)
M, ll, theta, U_indiv, _ = M.fit_em_ninits(iter=200, tol=0.01, fit_arrangement=False,
                                           fit_emission=True, init_arrangement=True,
                                           init_emission=True, n_inits=50, first_iter=30,
                                           verbose=False)

# M.initialize([data])
# U_indiv, _ = M.Estep()

# printing kappa
print(f'kappa: {M.emissions[0].kappa}')

# saving U and U_indiv
output['U'] = U
output['U_indiv'] = U_indiv
# output['M'] = M
output['ll'] = ll
# output['theta'] = theta

with open(PKL_output, 'wb') as pf:
    pickle.dump(output, pf)

# loading saved U and U_indiv
with open(PKL_output, 'rb') as pf:
    output = pickle.load(pf)

U = output['U']
U_indiv = output['U_indiv']
ll = output['ll']

# Load colormap and labels
# lid,cmap,names = nt.read_lut('atl-NettekovenSym32.lut')
lid,cmap,names = nt.read_lut(os.path.join(surface_helpers_dir, 'atl-glasser.lut'))

flat_surf_L = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'fs_LR.32k.R.flat.surf.gii')
border_LR = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.border')

def plot_probseg(surf_data, cmap, hemi):
    label = np.argmax(surf_data, axis=0)+1
    [label_L, label_R] = surf_from_cifti(atlas.data_to_cifti(label.reshape(1, -1)))

    if hemi == 'L':
        # left cortex
        flatmap.plot(label_L.reshape(-1, ),
                     surf=flat_surf_L,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     label_names=names,
                     new_figure=False,
                     frame=None,
                     render='matplotlib',
                     cmap=cmap,
                     borders=border_LR,
                     # cscale=[0,31],
                     overlay_type='label',
                     bordersize=3,
        )

    else:
        # right cortex
        flatmap.plot(label_R.reshape(-1, ),
                     surf=flat_surf_R,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     label_names=names,
                     new_figure=False,
                     frame=None,
                     render='matplotlib',
                     cmap=cmap,
                     borders=border_LR,
                     # cscale=[0, 31],
                     overlay_type='label',
                     bordersize=3,
        )


# Make a nifti image of the first subject
surf_data = U.detach().numpy()

# plot the group probabilistic atlas
plt.figure()
plt.subplot(1, 2, 1)
plot_probseg(surf_data, cmap, 'L')
plt.subplot(1, 2, 2)
plot_probseg(surf_data, cmap, 'R')
plt.suptitle('group')

# plot 3 individual subjects
for i,s in enumerate([6,9,12]):
    surf_data = U_indiv[s].detach().numpy()

    plt.figure()
    plt.subplot(1, 2, 1)
    plot_probseg(surf_data, cmap, 'L')
    plt.subplot(1, 2, 2)
    plot_probseg(surf_data, cmap, 'R')
    plt.suptitle(f'subject {s}')


# inspect model training
plt.figure(figsize=(5,5))
plt.plot(ll)

JPG_fig = os.path.join(resultsPath, f'example_parcellations_{appendix}.jpg')
plt.savefig(JPG_fig, format='jpg', dpi=400)

pass
