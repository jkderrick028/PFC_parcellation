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
import SUITPy.flatmap as flatmap
import torch
import SUITPy as suit


projectPath, mainResultsPath = setProjectPath()

dataset_name = 'MDTB' # or Demand

resultsPath = os.path.join(mainResultsPath, os.path.basename(__file__).replace('.py', ''), dataset_name)
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

appendix = 'cerebellum'

PKL_output = os.path.join(resultsPath, f'output_{appendix}.pkl')
output = {}

## loading MDTB data
PKL_data = os.path.join(projectPath, 'data', f'{dataset_name}_Cond_All_ses-s1_MNISymC3.pkl')
with open(PKL_data, 'rb') as pf:
    original_data = pickle.load(pf)
    data = original_data['X_individuals']
    info_individuals = original_data['info_individuals']
    dataset_obj_individuals = original_data['dataset_obj_individuals']

cond_vec = np.array(list(info_individuals[dataset_obj_individuals.cond_ind]))
part_vec = np.ones(len(cond_vec)).astype(int)

## loading group atlas
atlas, _ = am.get_atlas('MNISymC3')
atlas_fname = os.path.join(surface_helpers_dir, 'atl-NettekovenSym32_space-MNI152NLin2009cSymC_probseg.nii.gz')
U = atlas.read_data(atlas_fname)

# Build the arrangement model - the parameters are the log-probabilities of the atlas
# ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas)
ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas)

# fit the emission model to the data
# K is the number of parcels
K = ar_model.K
# Make a design matrix
X= ut.indicator(cond_vec)
# Build an emission model
# em_model = em.MixVMF(K=K,P=atlas.P, X=X,part_vec=part_vec)
em_model = em.MixVMF(K=K, P=U.shape[1], X=X, part_vec=part_vec)

# Build the full model: The emission models are passed as a list, as usually we have multiple data sets
M = fm.FullMultiModel(ar_model, [em_model])
# Attach the data to the model - this is done for speed
# The data is passed as a list with on element per data set
M.initialize([data])

# Now we can run the EM algorithm
M, ll, _, U_indiv = M.fit_em(iter=1000, tol=0.01, fit_arrangement=False,fit_emission=True,first_evidence=False)

# get the data only parcellation
emloglik  = M.emissions[0].Estep()
Uhat_data = torch.softmax(emloglik, dim=1)

# printing kappa
print(f'kappa: {M.emissions[0].kappa}')

# saving U and U_indiv
output['U'] = U
output['U_indiv'] = U_indiv
output['Uhat_data'] = Uhat_data
# output['M'] = M
output['ll'] = ll
output['V'] = M.emissions[0].V 
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

# Load colormap and labels
lid,cmap,names = nt.read_lut(os.path.join(surface_helpers_dir, 'atl-NettekovenSym32.lut'))


def plot_probseg(nifti, cmap):
    surf_data = suit.flatmap.vol_to_surf(nifti, stats='nanmean', space='MNISymC')
    label = np.argmax(surf_data, axis=1) + 1
    flatmap.plot(label.reshape(-1, ),
                 alpha=1,
                 label_names=names,
                 new_figure=False,
                 frame=None,
                 render='matplotlib',
                 cmap=cmap,
                 overlay_type='label',
                 bordersize=3,
                 undermap='gray',
                 underscale=[-1, 0.5]
    )


# plot the group probabilistic atlas
plt.figure()
plt.subplot(1, 4, 1, title='group')

# Make a nifti image of the first subject
nifti = atlas.data_to_nifti(U)
plot_probseg(nifti, cmap)


# plot 3 individual subjects
for i,s in enumerate([6,9,12]):
    plt.subplot(1, 4, i+2, title=f'subject {s}')
    nifti = atlas.data_to_nifti(U_indiv[s].numpy())
    plot_probseg(nifti, cmap)


JPG_fig = os.path.join(resultsPath, f'example_parcellations_{appendix}.jpg')
plt.savefig(JPG_fig, format='jpg', dpi=400)

# inspect model training
plt.figure(figsize=(5,5))
plt.plot(ll)

JPG_fig = os.path.join(resultsPath, f'll_training_{appendix}.jpg')
plt.savefig(JPG_fig, format='jpg', dpi=400)

pass
