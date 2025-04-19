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

projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

# Get the atlas
atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)

# Sample the probabilistic atlas at the specific atlas grayordinates
# atlas_fname = os.path.join(surface_helpers_dir, 'atl-NettekovenSym32_space-MNI152NLin2009cSymC_probseg.nii.gz')
atlas_fname = [os.path.join(surface_helpers_dir, 'glasser.L.label.gii'), os.path.join(surface_helpers_dir, 'glasser.R.label.gii')]
U = atlas.read_data(atlas_fname)
U = U.T

# converting the hard parcellation into a probabilistic one
U = IndividualParcellation.utils.convert_hard_to_prob(U, strength=7.0)

# Build the arrangement model - the parameters are the log-probabilities of the atlas
# ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas)
ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas)

# loading MDTB data
PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_Half.pkl')
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
M, _, _, _ = M.fit_em(iter=200, tol=0.01,
    fit_arrangement=False,fit_emission=True,first_evidence=False)

M.initialize([data])
U_indiv, _ = M.Estep()

def plot_probseg(nifti,cmap):
    # Project the nifti image to the surface over the MNISymC space
    surf_data = suit.flatmap.vol_to_surf(nifti, stats='nanmean',space='MNISymC')
    label = np.argmax(surf_data, axis=1)+1

    suit.flatmap.plot(label,
        render='matplotlib',
        cmap=cmap,
        cscale=[0,31],
        label_names = names,
        new_figure=False,
        overlay_type='label',
        bordersize=3,
    )


# Load colormap and labels
# lid,cmap,names = nt.read_lut('atl-NettekovenSym32.lut')
lid,cmap,names = nt.read_lut('atl-glasser.lut')

# Make a nifti image of the first subject
nifti = atlas.data_to_nifti(U)

# Make a figure
plt.figure(figsize=(20,5))

# plot the group probabilistic atlas
plt.subplot(1,4,1,title='group')
plot_probseg(nifti,cmap)

# plot 3 individual subjects
for i,s in enumerate([6,9,12]):
    plt.subplot(1,4,i+2,title=f'subject {s}')
    nifti = atlas.data_to_nifti(U_indiv[s].numpy())
    plot_probseg(nifti,cmap)

pass
