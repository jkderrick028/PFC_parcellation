import os.path
import numpy as np
import Functional_Fusion.atlas_map as am
from py_util_dx.py_utils import setProjectPath
from nitools.cifti import surf_from_cifti
import nitools as nt
import SUITPy.flatmap as flatmap
import similarity_colormap as sc
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd


projectPath, mainResultsPath = setProjectPath()

surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

flat_surf_L = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.flat.surf.gii')
flat_surf_R = os.path.join(surface_helpers_dir, 'fs_LR.32k.R.flat.surf.gii')
border_LR = os.path.join(surface_helpers_dir, 'fs_LR.32k.L.border')

atlas_str = 'fs32k'
atlas, ainf = am.get_atlas(atlas_str)


def plot_parcel_size(Prob, cmap, labels, wta=True, sort=True, side=None):
    sumP, sumV = calc_parcel_size(Prob)

    D = pd.DataFrame({'region': labels[1:],
                      'sumP': sumP,
                     'sumV': sumV,
                      'cnum': np.arange(Prob.shape[0]) + 1})
    if sort:
        D = D.sort_values(by='region')
    if side is not None:
        D['side'] = D.region.str[-1]
        D = D[D.side == side]
    pal = {d.region: cmap(d.cnum) for i, d in D.iterrows()}
    if wta:
        sb.barplot(data=D, y='region', x='sumV', palette=pal)
    else:
        sb.barplot(data=D, y='region', x='sumP', palette=pal)
    return D


def calc_parcel_size(Prob):
    """Calculates probabilstic and winner-take all cluster size
    from probabilities

    Args:
        Prob (ndarray):
    returns:
        sumP: sum of probabilities
        sumV: number of hard-assigned voxels
    """
    if isinstance(Prob, pt.Tensor):
        Prob = Prob.numpy()
    if Prob.ndim == 3:
        voxel_axis = 2
    else:
        voxel_axis = 1
    parcel_axis = voxel_axis - 1
    sumP = np.sum(Prob, axis=voxel_axis)
    counts = np.zeros(Prob.shape)
    if Prob.ndim == 2:
        counts[np.argmax(Prob, axis=parcel_axis), np.arange(Prob.shape[voxel_axis])] = 1
    else:
        # Loop over subjects to get voxel counts for each subject
        for sub in np.arange(0, Prob.shape[0]):
            counts[sub, np.argmax(Prob, axis=parcel_axis)[sub], np.arange(Prob.shape[voxel_axis])] = 1
    sumV = np.sum(counts, axis=voxel_axis)
    return sumP, sumV


def parcel_similarity(model, plot=False, sym=False, weighting=None):
    """ Calculates a parcel similarity based on the V-vectors (functional profiles) of the emission models

    Args:
        model (FullMultiModel): THe model
        plot (bool, optional): Generate plot? Defaults to False.
        sym (bool, optional): Generate similarity in a symmetric fashion? Defaults to False.
        weighting (ndarray, optional): possible weighting of different dataset. Defaults to None.

    Returns:
        w_cos_sim: Weighted cosine similarity (integrated)
        cos_sim: Cosine similarity for each data set
        kappa: Kappa from each dataset(?)

    """
    n_sets = len(model.emissions)
    if sym:
        K = int(model.emissions[0].K / 2)
    else:
        K = model.emissions[0].K
    cos_sim = np.empty((n_sets, K, K))
    if model.emissions[0].uniform_kappa:
        kappa = np.empty((n_sets,))
    else:
        kappa = np.empty((n_sets, K))
    n_subj = np.empty((n_sets,))

    V = []
    for i, em in enumerate(model.emissions):
        if sym:
            # Average the two sides for clustering
            V.append(em.V[:, :K] + em.V[:, K:])
            V[-1] = V[-1] / np.sqrt((V[-1]**2).sum(axis=0))
            if model.emissions[0].uniform_kappa:
                kappa[i] = em.kappa
            else:
                kappa[i] = (em.kappa[:K] + em.kappa[K:]) / 2
        else:
            V.append(em.V)
            kappa[i] = em.kappa
        cos_sim[i] = V[-1].T @ V[-1]

        # V is weighted by Kappa and number of subjects
        V[-1] = V[-1] * np.sqrt(kappa[i] * em.num_subj)
        if weighting is not None:
            V[-1] = V[-1] * np.sqrt(weighting[i])

    # Combine all Vs and renormalize
    Vall = np.vstack(V)
    Vall = Vall / np.sqrt((Vall**2).sum(axis=0))
    # Calculate similarity
    w_cos_sim = Vall.T @ Vall

    # Integrated parcel similarity with kappa
    if plot is True:
        plt.figure()
        grid = int(np.ceil(np.sqrt(n_sets + 1)))
        for i in range(n_sets):
            plt.subplot(grid, grid, i + 1)
            plt.imshow(cos_sim[i, :, :], vmin=-1, vmax=1)
            plt.title(f"Dataset {i+1}")
        plt.subplot(grid, grid, n_sets + 1)
        plt.imshow(w_cos_sim, vmin=-1, vmax=1)
        plt.title(f"Merged")

    return w_cos_sim, cos_sim, kappa


def colour_parcel(mname, sym=False, labels=None, clusters=None, gamma=0):
    """
    Colours the parcellation of a model.

    Args:
    - mname (str): Path of the model to be analyzed.
    - sym (bool): Whether to generate similarity in a symmetric fashion. Defaults to True.
    - plot (bool): Whether or not to generate plots. Defaults to True.
    - labels (ndarray): Labels for the parcels if they have already been generated. Defaults to None.
    - clusters (ndarray): Distorts color towards cluster mean.
    - gamma (float): The gamma value used for the colormap.

    Returns:
    - Prob (ndarray): The winner-take-all probabilities for each region.
    - parcel (ndarray): The parcel label for each region.
    - atlas (object): The atlas object used for the parcellation.
    - labels (ndarray): The labels for the clusters generated by clustering.
    - cmap (object): The colormap generated for the parcellation.
    """

    # Get model and atlas.
    fileparts = mname.split("/")
    split_mn = fileparts[-1].split("_")
    info, model = ut.load_batch_best(mname)
    atlas, ainf = am.get_atlas(info.atlas, ut.atlas_dir)

    # Get winner-take all parcels
    Prob = np.array(model.arrange.marginal_prob())
    parcel = Prob.argmax(axis=0) + 1

    # Make a colormap.
    w_cos_sim, _, _ = parcel_similarity(model, plot=False, sym=sym)
    W = sc.calc_mds(w_cos_sim, center=True)
    if sym:
        W = np.concatenate([W, W])
    # Define color anchors
    m, regions, colors = sc.get_target_points(atlas, parcel)
    cmap = sc.colormap_mds(
        W, target=(m, regions, colors), clusters=clusters, gamma=gamma
    )
    sc.plot_colorspace(cmap(np.arange(model.K)))

    plt.figure(figsize=(5, 10))
    plot_parcel_size(Prob, cmap, labels, wta=True)

    return Prob, parcel, atlas, labels, cmap



def plot_flatmap_labels(label, hemi, frame=None, borders=True, atlas='glasser'):
    # lid, cmap, names = nt.read_lut(os.path.join(surface_helpers_dir, f'atl-{atlas}.lut'))
    df = pd.read_csv(os.path.join(surface_helpers_dir, f'atl-{atlas}.lut'), header=None, names=['lid', 'r', 'g', 'b', 'roi'], sep='\s+', dtype={'r': float, 'g': float, 'b': float, 'lid': int})
    cmap = df[['r', 'g', 'b']].to_numpy()
    names = df['roi'].tolist()

    # if atlas == 'schaefer100':
    #     cmap = cmap[np.arange(1, len(cmap), 2)] / 255
    #     names = list(lid[np.arange(0, len(lid), 2)])

    if hemi == 'L':
        # left cortex
        if atlas == 'schaefer100':
            keep_inds = np.arange(len(cmap))
        else:
            keep_inds = np.arange(int(cmap.shape[0]/2))
        cmap = np.vstack([np.ones((1, 3)), cmap[keep_inds, :], np.ones((1, 3))])

        flatmap.plot(label.reshape(-1, ),
                     surf=flat_surf_L,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     label_names=list(map(names.__getitem__, keep_inds)),
                     new_figure=False,
                     frame=frame,
                     render='matplotlib',
                     cmap=cmap,
                     borders=None if borders is None else border_LR,
                     overlay_type='label',
                     bordersize=3,
                     undermap='gray',
                     underscale=[-1, 0.5]
        )

    else:
        # right cortex
        if atlas == 'schaefer100':
            keep_inds = np.arange(len(cmap))
        else:
            keep_inds = np.arange(int(cmap.shape[0]/2), cmap.shape[0])
        cmap = np.vstack([np.ones((1, 3)), cmap[keep_inds, :], np.ones((1, 3))])

        flatmap.plot(label.reshape(-1, ),
                     surf=flat_surf_R,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     label_names=list(map(names.__getitem__, keep_inds)),
                     new_figure=False,
                     frame=frame,
                     render='matplotlib',
                     cmap=cmap,
                     borders=None if borders is None else border_LR,
                     overlay_type='label',
                     bordersize=3,
                     undermap='gray',
                     underscale=[-1, 0.5]
        )


def flatmap_real_vals(surf_data, hemi, cscale=[-1, 1], cmap='bwr', colorbar=False):
    [label_L, label_R] = surf_from_cifti(atlas.data_to_cifti(surf_data.reshape(1, -1)))

    if hemi == 'L':
        # left cortex
        flatmap.plot(label_L.reshape(-1, ),
                     surf=flat_surf_L,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.L.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     new_figure=False,
                     frame=None,
                     cmap=cmap,
                     borders=border_LR,
                     bordersize=1,
                     cscale=cscale
        )

    else:
        # right cortex
        flatmap.plot(label_R.reshape(-1, ),
                     surf=flat_surf_R,
                     underlay=os.path.join(surface_helpers_dir, 'sub-01.R.sulc.32k_fs_LR.shape.gii'),
                     alpha=1,
                     new_figure=False,
                     frame=None,
                     cmap=cmap,
                     borders=border_LR,
                     bordersize=1,
                     cscale=cscale,
                     colorbar=colorbar
        )
