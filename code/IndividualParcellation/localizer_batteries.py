#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compares predictive performance of different localizer batteries
Author: Caro Nettekoven
"""

import pandas as pd
from pathlib import Path
import numpy as np
import torch as pt
import matplotlib.pyplot as plt

from copy import copy,deepcopy
import Functional_Fusion.dataset as ds
import ProbabilisticParcellation.util as ut 
import ProbabilisticParcellation.evaluate as ppev
import ProbabilisticParcellation.individ_group as ig
import nitools as nt
import Functional_Fusion.atlas_map as am
import IndividualParcellation.scripts.paths as paths
import ProbabilisticParcellation.scripts.feature_model as fm
from ProbabilisticParcellation.scripts.ridge_reg import ridgeFit
from ProbabilisticParcellation import functional_profiles as fp
from sklearn.decomposition import PCA



def domain_masks(atlas = 'NettekovenSym32', space='MNISymC2', select_masks = None):
    # Set paths
    atlas_dir = paths.set_atlas_dir() 
    space, _ = am.get_atlas(space)
    if 'MNISym' in space.name:
        space_folder = 'tpl-MNI152NLin2009cSymC'

    # Get domain parcellation
    parcellation = f'/{space_folder}/atl-{atlas}_space-{space_folder.split("tpl-")[1]}_probseg.nii'
    pseg = space.read_data(atlas_dir + parcellation)
    pseg = pseg.T

    # Get domain labels
    _, cmap, labels = nt.read_lut(f'{atlas_dir}/{space_folder}/atl-{atlas}.lut') 

    dseg = np.argmax(pseg, axis=0) + 1
    mask_names = ['Motor', 'Action', 'Demand', 'SLS']
    masks = {}
    for name in mask_names:
        masks[name] = np.isin(dseg, [l for l, label in enumerate(labels) if label[0] == name[0]])

    # Get whole cerebellar mask
    masks['Cerebellum'] = np.ones_like(dseg).astype(bool)

    # Get masks for domain boundaries
    masks['Motor-Action'] = masks['Motor'] | masks['Action']
    masks['Demand-SLS'] = masks['Demand'] | masks['SLS']
    masks['Motor-Demand'] = masks['Motor'] | masks['Demand'] | masks['Action']

    if select_masks == 'combinations':
        select_masks = ['Cerebellum', 'Demand-SLS', 'Motor-Demand']

    if select_masks is not None:
        masks = {k: masks[k] for k in select_masks}

    return masks

def feature_matrix(profile_name='NettekovenSym32_profile_individ_MDTB-s1'):
    """Function to load the feature matrix for a given atlas and MDTB sessions"""
    # Load the functional profile
    Data = pd.read_csv(ut.model_dir + '/Atlases/Profiles/' +
                    profile_name + '.tsv', delimiter='\t')

    _, cmap, regions = nt.read_lut(ut.model_dir + '/Atlases/' +
                                    profile_name.split('_')[0] + '.lut')
    regions = regions[1:]

    Data = Data[['condition'] + regions]
    tags = fm.load_features()
    tags_individ = fm.subject_features(tags, Data)
    task_matrix, task_codes = fm.task_indicator(Data)
    tags_task = np.concatenate(
        (tags_individ.T.to_numpy(), task_matrix), axis=1)

    Data_norm, tags_norm = fm.normalize(Data[regions], tags_task)

    # Ridge regression
    R2, features = ridgeFit(Data_norm.to_numpy(), tags_norm,
                            fit_intercept=False, voxel_wise=False, alpha=1.0)
    
    # Make dataframe
    Features = pd.DataFrame(features.T, columns=[
        'left_hand', 'right_hand', 'saccades'] + list(task_codes.keys()), index=regions)


    # Replace task names with the names used in King et al. (2019) to describe the tasks for consistency
    replace = {'SpatialNavigation': 'SpatialImagery',
                'RomanceMovie': 'AnimatedMovie', 'VideoAct': 'VideoActions'}
    for i in range(len(Features.columns.tolist())):
        if Features.columns.tolist()[i] in replace.keys():
            Features.rename(columns={Features.columns.tolist()[i]: replace[Features.columns.tolist()[i]]}, inplace=True)  
    
    
    return Features

def plot_feature_matrix(Features, save=False):
    """Function to plot the feature matrix of the functional profiles
    Args:
        Features (pd.DataFrame): Functional profiles of K regions and N tasks (K x N matrix). Each entry is the activation of a region for a task.
        save (bool): Whether to save the figure
    """
    regions = Features.index.tolist()
    # Plot
    cmap = plt.get_cmap('RdBu_r')
    plt.figure(figsize=(20, 10))
    plt.imshow(Features, cmap=cmap)
    plt.yticks(np.arange(len(regions)), regions)
    plt.xticks(np.arange(len(Features.columns.tolist())),
            Features.columns.tolist(), rotation=90)

    # Plot a horizontal line in the middle
    plt.hlines(len(regions) / 2 - 0.5, 0,
            len(Features.columns.tolist()), color='black', linewidth=2)
    if save:
        plt.savefig(f'{ut.figure_dir}/feature_matrix_s1.pdf', dpi=300)

        # Save functional profiles controlling for motor responses
        Features.to_csv(ut.figure_dir + 'functional_profile_s1.csv', sep='\t')


def distinct_tasks(functional_profile):
    """Function to select most distinct tasks from the functional profiles
    Args:
        functional_profile (pd.DataFrame): Functional profiles of K regions and N tasks (K x N matrix). Each entry is the activation of a region for a task.
        
    Returns:
        tasks (np.array): List of tasks that are best able to distinguish between regions"""
    pca = PCA()
    pca.fit(functional_profile)

    # Calculate correlation between columns and principal components
    correlation_matrix = np.abs(np.corrcoef(functional_profile, pca.components_.T))

    # Extract correlation values for each column
    column_correlation = correlation_matrix[:functional_profile.shape[1], functional_profile.shape[1]:]

    # Select the columns with highest absolute correlation
    ordered_columns = np.argsort(-np.max(column_correlation, axis=1))

    tasks = functional_profile.columns[ordered_columns]
    return tasks


def build_localizer(functional_profile, max_duration=20, regions='all'):
    """Builds a localizer for a given set of regions"""

    if regions == 'all':
        regions = functional_profile.index.tolist()
    # Get most distinct tasks for given regions
    tasks = distinct_tasks(functional_profile.loc[regions])

    # Import the task durations and build localizer
    localizer = {}
    length = 0
    while length < max_duration:
        for task in tasks:
            if length < max_duration:
                localizer.append(task)
                length += durations[durations['task']==task]['duration'].values[0]
    return localizer
    # # Fill 20 minutes of task batteries with tasks that are most distinct between regions of one domain
    # localizer_batteries = {}
    # length_battery = 20
    # # Import the task durations
    # data_dir = paths.set_fusion_dir()
    # durations = pd.read_csv(f'{data_dir}/../Cerebellum/super_cerebellum/sc1_sc2_taskConds_conn.txt', delimiter='\t')
    # while length_battery > 0:
    #     for domain, tasks in domain_tasks.items():
    #         if length_battery > 0:
    #             localizer_batteries[domain] = []
    #             for task in tasks:
    #                 if length_battery > 0:
    #                     localizer_batteries[domain].append(task)
    #                     length_battery -= durations[durations['task']==task]['duration'].values[0]




if __name__ == "__main__":
    # Import atlas labels
    atlas = 'NettekovenSym32'
    atlas_dir = paths.set_atlas_dir()
    _, cmap, labels = nt.read_lut(f'{atlas_dir}/tpl-MNI152NLin2009cSymC/atl-{atlas}.lut')
    # Get the functional profile of only session 1 to avoid overfitting
    model_dir = paths.set_model_dir()
    profile_name = f'{atlas}_profile_individ_MDTB-s1'

    # Check if the functional profile exists, else create it
    fname = f'{model_dir}/../Atlases/Profiles/{profile_name}'
    if not Path(fname+ '.tsv').exists():
        mname = f"Models_03/{atlas}_space-MNISymC2"
        info, model = ut.load_batch_best(mname)
        info = ut.recover_info(info, model, mname)
        # Get only the functional profile of MDTB session 1
        info.datasets = ['MDTB']
        info.sess = [['ses-s1']]
        info.type = ['CondHalf']
        model.emissions = model.emissions[:1]
        model.emissions[0].parcel_specific_kappa = False
        model.emissions[0].subject_specific_kappa = False
        fp.export_profile(mname, info, model, labels, source="individ", fname=fname + '.tsv')

    Features = feature_matrix(profile_name)
    plot_feature_matrix(Features, save=False)
    # Remove left hand right hand and saccades as features
    Features.drop(columns=['left_hand', 'right_hand', 'saccades'], inplace=True)

    # List of region names
    domains = {'M': [], 'A': [], 'D': [], 'S': []}
    for region in labels[1:]:
        prefix = region[0]
        if prefix in domains:
            domains[prefix].append(region)

    motor = domains.get('M', [])
    action = domains.get('A', [])
    demand = domains.get('D', [])
    sls = domains.get('S', [])

    # --- Make localizers ---
    localizers = {}
    for domain, regions in domains.items():
        localizer = build_localizer(Features, regions=regions)
        localizers[domain] = localizer
        

    
    # # Evaluate localizer batteries
    # for key, localizer_tasks in localizer_batteries.items():
    #     # Evaluate two runs of a given localizer battery
    #     pass
    
