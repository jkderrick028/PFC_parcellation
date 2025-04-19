import numpy as np
import pandas as pd
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.special import softmax
from sklearn.linear_model import LinearRegression

import Functional_Fusion.Functional_Fusion.atlas_map as am
import Functional_Fusion.Functional_Fusion.dataset as ds
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.emissions as emi
import IndividualParcellation.scripts.dual_regression as dr
import IndividualParcellation.scripts.dictionary_learning as dl
import FusionModel.evaluate as ev
import FusionModel.util as futil

from IndividualParcellation.global_config import MODEL_DIR, BASE_DIR, HOME_DIR, DEVICE



#################### FUNCTIONS MODIFIED TO USE HBP V ####################


def get_indiv_parcellation(ar_model, atlas, train_data, cond_vec, part_vec,
                           subj_ind, Vs=None, sym_type='asym', n_iter=200,
                           uniform_kappa=True, fit_arrangement=False,
                           fit_emission=True, device=None):
    
    # convert tdata to tensor
    if type(train_data) is np.ndarray:
        train_data = pt.tensor(train_data, dtype=pt.get_default_dtype())
    if Vs is None:
        Vs = [None] * len(train_data)

    # Check if the lists have equal length using assert
    assert len(train_data) == len(cond_vec) == len(part_vec) == len(Vs),\
        "training data, condition vector, and partition vector " \
        "must have equal length."

    # Check if the input arrangement model is valid
    if not isinstance(ar_model, ar.ArrangementModel):
        raise ValueError("The input model must be a valid arrangement"
                         " model object")

    # Initialize emission models
    em_models = []
    for j, this_cv in enumerate(cond_vec):
        em_model = emi.build_emission_model(ar_model.K if sym_type=='asym'
                                            else ar_model.K_full,
                                            atlas, 'VMF',
                                            fm.indicator(this_cv), None,
                                            V=Vs[j])
        em_models.append(em_model)

    M = fm.FullMultiModel(ar_model, em_models)
    M.initialize(train_data, subj_ind=subj_ind)

    if M.arrange.name.startswith('indp'):
        M, ll, _, U_full = M.fit_em(iter=n_iter, tol=0.01,
                                     fit_arrangement=fit_arrangement,
                                     fit_emission=fit_emission,
                                     first_evidence=False)
    else:
        raise NameError("The arrangement model is not supported yet.")
    
    emloglik = M.emissions[0].Estep()
    U_data = pt.softmax(emloglik, dim=1)

    # Return the individual PROBABILISTIC parcellations
    return U_full, U_data, M.emissions[0].V, M.emissions[0].X, M


#################### PIPELINE FUNCTIONS ####################


def split_dataset(data, info, itrain, itest, sess=True):
    
    train, test, cond, part, sub = [], [], [], [], []
    
    sessions = np.unique(info.sess)
    runs = np.unique(info.study)
    cond_num = info.cond_num_uni.values
    part_num = info.study.values
    n_sub = data.shape[0]
    
    for i in itrain:
        idx = (info.sess == sessions[i]) if sess else (info.half == runs[i])
        train.append(data[:,idx])
        cond.append(cond_num[idx].reshape(-1,))
        part.append(part_num[idx].reshape(-1,))
        sub.append(np.arange(0, n_sub))


    for i in itest:
        idx = info.sess == sessions[i]
        test.append(data[:,idx])
    
    inf = pd.DataFrame({'subj_num': np.arange(0, n_sub)})
    inf['train_set'] = [itrain] * 24
    inf['test_set'] = [itest] * 24
            
    return train, test, cond, part, sub, inf


def get_indiv_parcels(model, U, tdata, inf, sdl_method='online', sdl_init='kmeans', V=None, X=None):
        
    if model == 'DR':
        parcels = dr.get_iparcel_dualreg(tdata[0], Ug=U, V=V)
    elif model == 'SDL':
        parcels,_,_ = dl.get_iparcel_dictlearning(tdata[0], n_parcels=17, vinit_type=sdl_init,
                                              dict_init=V, method=sdl_method, alpha=.01,
                                              l1_ratio=.5, write_dir='/tmp')
    parcels = pt.tensor(parcels)

    inf_copy = inf.copy()
    inf_copy['method'] = model

    return parcels, inf_copy


def evaluate_parcels(U, indiv_parcels, dtest, atlas, inf, inf_col='DCBC'):

    # distance matrix
    dist = ev.compute_dist(atlas.world.T, resolution=1)

    if type(dtest) is np.ndarray:
        dtest = pt.tensor(dtest)

    dim = 1 if indiv_parcels.dim() == 3 else 0
    pindiv = pt.argmax(indiv_parcels, dim=dim) + 1
    dcbc = ev.calc_test_dcbc(pindiv, dtest, dist)
    inf[inf_col] = dcbc
    
    return dcbc, inf


def plot_multi_flat(data, atlas, grid, cmap='tab20b', dtype='label',
                    cscale=None, title=None, sub_titles=None, colorbar=False,
                    save_fig=False, save_fname=None):

    if isinstance(data, np.ndarray):
        n_subplots = data.shape[0]
    elif isinstance(data, list):
        n_subplots = len(data)
    elif pt.is_tensor(data):
        data = data.numpy()
        n_subplots = data.shape[0]

    if not isinstance(cmap, list):
        cmap = [cmap] * n_subplots

    for i in np.arange(n_subplots):
        plt.subplot(grid[0], grid[1], i + 1)
        futil.plot_data_flat(data[i], atlas,
                       cmap=cmap[i],
                       dtype=dtype,
                       cscale=None,
                       render='matplotlib',
                       colorbar=(i == 0) & colorbar)

        plt.title(sub_titles[i])
        plt.tight_layout()
    plt.title(title)

    if save_fig:
        plt.savefig(save_fname)


#################### DATA PREP ####################
# load atlas
atlas, _ = am.get_atlas('MNISymC3')
# load group prior
model_name = f'/Models_03/asym_Md_space-MNISymC3_K-17'
U, minfo = ar.load_group_parcellation(MODEL_DIR + model_name, device='cpu')

# load individual data
npyfile = 'data/tdata_mdtb_condall_ses-all.npy'
data = np.load(HOME_DIR + npyfile)
# load info file
infofile = 'data/tinfo_mdtb_condall_ses-all.tsv'
info = pd.read_csv(HOME_DIR + infofile, sep='\t')

'''data, info, _ = ds.get_dataset(HOME_DIR, 'data/tdata_mdtb_condall_ses-all.npy', atlas=atlas.name,
                               subj=None, sess='ses-s1',
                               type='CondAll')'''

# SDL parameters
alpha = .01
l1_ratio = .5

#################### PIPELINE ####################

# split MDTB by session
# train on s1, test on s2
dtrain, dtest, cond_v, part_v, sub_i, info_df = split_dataset(data, info, [0], [1], sess=True)

# HBP parcellations with fixed V
ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas, sym_type='asym')
Vs, _ = emi.load_emission_params(MODEL_DIR + '/Models_03/asym_Md_space-MNISymC3_K-17', 'V', device='cpu')
parcels_hbp_full, parcels_hbp_data, V, X, M = get_indiv_parcellation(ar_model, atlas, dtrain, cond_v, part_v, sub_i, Vs=[Vs[0]])
hbp_df = info_df.copy()
hbp_df['method'] = 'HBP'

# DR parcellations
U_ica = dr.group_ica(dtrain[0], n_components=17) # group ICA
y = pt.matmul(pt.linalg.pinv(X), pt.tensor(dtrain, dtype=pt.get_default_dtype()))
parcels_dr_uica, dr_df = get_indiv_parcels('DR', U_ica, y, info_df, V=Vs[0])

# SDL parcellations
y = pt.matmul(pt.linalg.pinv(X), pt.tensor(dtrain, dtype=pt.get_default_dtype()))
parcels_sdl_sparse, sdl_df = get_indiv_parcels('SDL', U.T, y, info_df, sdl_method='sparse', sdl_init='hbp', V=Vs[0], X=X)

# evaluations
eval_hbp_full, hbp_df = evaluate_parcels(U, parcels_hbp_full, dtest[0], atlas, hbp_df, 'DCBC_data_group')
eval_hbp_data, hbp_df = evaluate_parcels(U, parcels_hbp_data, dtest[0], atlas, hbp_df, 'DCBC_data')
eval_dr, dr_df = evaluate_parcels(U, parcels_dr_uica, dtest[0], atlas, dr_df, 'DCBC_DR')
eval_sdl_sparse, sdl_df = evaluate_parcels(U, parcels_sdl_sparse, dtest[0], atlas, sdl_df, 'DCBC_sparse')
eval_group, hbp_df = evaluate_parcels(U, U, dtest[0], atlas, hbp_df, 'DCBC_group') # group map evaluation

# plot DCBC
dcbc_df = pd.DataFrame({'Group': hbp_df.DCBC_group,
                   'HBP (group+data)': hbp_df.DCBC_data_group,
                   'HBP (data only)': hbp_df.DCBC_data,
                   'DR (NNLS)': dr_df.DCBC_GICA_U,
                   'SDL (sparse)': sdl_df.DCBC_sparse})
plt.subplots(figsize=(15,6))
plt.ylabel('DCBC')
sb.barplot(dcbc_df)

#################### OPTIONAL - PLOT FLATMAPS ####################

plt.figure(figsize=(20,20))
parcels = [parcels_hbp_full, parcels_hbp_data, parcels_dr, parcels_dr_v]
labels = ['HBP_group_data', 'HBP_data_only)----', 'DR', 'DR _HBP_V)']
for i,parc in enumerate(parcels):
    plot_multi_flat(parc,
                'MNISymC3', grid=(6,4),
                cmap='tab20', dtype='prob',
                title=f'{labels[i]} Individual Parcellations',
                sub_titles=["subj_{}".format(i+1) for i in range(parc.shape[0])],
                save_fig=True,
                save_fname=f'iparcels_{labels[i]}.png')