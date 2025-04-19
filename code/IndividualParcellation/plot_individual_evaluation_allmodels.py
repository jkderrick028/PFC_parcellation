"""
DCBC evaluation of individual parcellations obtained from 
HBP, Dual Regression and Dictionary-Learning models

authors: Ana Luisa Pinho, Jennifer Yoon
emails: agrilopi@uwo.ca, jyoon94@uwo.ca

created: October, 2024
last update: December, 2024

Compatibility: Python 3.9.20
"""

from pathlib import Path

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statannotations.Annotator import Annotator
from scipy.stats import bootstrap, ttest_rel
from global_config import DEVICE


# ###################### FUNCTIONS #####################################

def plot_evaluation(df, p_values, online, sparse, scialpha, outpath, 
                    evaluation='dcbc', do_annotations=False):
    # Set figure and subplot
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.subplots_adjust(left=.125, right=.975, bottom=.1, top=.925)
    # Set colors
    blues = sns.color_palette(palette='Blues')
    oranges = sns.color_palette(palette='Oranges')
    greens = sns.color_palette(palette='Greens')
    custom_palette = [blues[3], blues[5], blues[2], oranges[2], oranges[4], 
                      greens[2], greens[4]]
    sns.barplot(data=df, errorbar=None, palette=custom_palette, alpha=.8)
    # Change location of bars
    shifts = np.array([.15, 0., -.15, 0., -.15, 0., -.15]) - .15
    xshifts = {key: value for key, value in zip(df.columns.values, shifts)}
    newx_pos = []
    for b, bar in enumerate(ax.patches):
        current_width = bar.get_width()
        new_x = bar.get_x() + xshifts[df.columns.values[b]]
        bar.set_x(new_x)
        newx_pos.append(new_x + current_width / 2)
    # Set xticks positions and labels with newline characters
    x_labels = ['HBP G', 'HBP G+I', 'HBP I', 'DR', 'NNLS', 'DL', 
                'Sparse']
    plt.xticks(newx_pos, x_labels)
    # Fontsize of x-labels
    plt.xticks(fontsize=26)
    # X-Label Padding
    ax.tick_params(axis='x', pad=10)
    # Fontsize of y-labels
    plt.yticks(fontsize=26)
    # Set the limits of the x-axis
    plt.xlim(-.75, 6.3)
    
    if evaluation == 'dcbc':
        # Set the limits of the y-axis
        plt.ylim(0., .25)
        # Set y-label
        plt.ylabel('DCBC', fontsize=26, labelpad=10)
    else:
        assert eval_metric[:6] == 'cosine'
        # Set the limits of the y-axis
        plt.ylim(0., 1.2)
        # Set y-label
        plt.ylabel('Cosine Error', fontsize=26, labelpad=10)

    # Add error bar manually at the new x positions
    for x, y_col in zip(newx_pos, df.columns.values):
        y = df[y_col].values
        y = (y,)
        bootstrap_results = bootstrap(y, np.mean, confidence_level=.95, 
                                      n_resamples=1000)
        confidence_interval = bootstrap_results.confidence_interval
        error = confidence_interval.high - confidence_interval.low
        plt.errorbar(x, np.mean(y), yerr=error, fmt='none', color='black', 
                     capsize=5, elinewidth=4, capthick=4)
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    # Annotations
    pairs = [('HBP G+I','HBP G'), 
             ('HBP G+I', 'HBP I'),
             ('HBP G', 'HBP I'),
             ('HBP I', 'DR'), 
             ('DR', 'NNLS'), 
             ('DR', online),
             ('HBP I', online),
             ('HBP I', sparse),
             (online, sparse)]
    # Convert wide to long format
    df_renamed = df.rename(columns={
        'HBP Group': 'HBP G', 'HBP Group + Individual': 'HBP G+I', 
        'HBP Individual': 'HBP I', 'DR GICA-U': 'DR', 'DL Online': 'DL'})
    long_df = pd.melt(df_renamed, var_name='Model', value_name='DCBC')
    annotator = Annotator(ax, pairs, data=long_df, x='Model', y='DCBC')
    annotator.configure(
        test=None,
        text_format="star", # text_format="simple"
        # test_short_name="pttest", # if former is "simple"
        fontsize=10., hide_non_significant=True)
    annotator.set_pvalues(p_values)
    annotator.configure(fontsize=26)
    if do_annotations:
        annotator.annotate()
    # Text
    fig.text(.81, .94, 'Error bar: 95% CI of the Mean', ha='center', 
             fontsize=24)
    fig.text(.23, .94, r'$\alpha = $' + scialpha, ha='center', fontsize=24)
    # Save figure
    plt.savefig(outpath)


# ####################### INPUTS #######################################
    
## Atlas
atlas_name = 'MNISymC3'
# atlas_fname = 'atl-NettekovenAsym32_space-MNI152NLin2009cSymC_probseg.nii.gz'
atlas_fname = 'asym_Md_space-MNISymC3_K-17_probseg.nii'
sym_type = 'asym'

# Dataset parameters
dataset = 'MDTB'

# Evaluation metrics
# eval_metric = 'dcbc'
# eval_metric = 'cosine_average'
eval_metric = 'cosine_expected'

# Paths
main_dir = Path.cwd()
eval_dir = Path(str(Path(main_dir, 'evaluation')))
atlas_path = Path(str(Path('atlases', atlas_fname)))

# ######################## RUN #########################################

if __name__ == "__main__":

    # What alpha for online and sparse to pick?

    dl_onlines = ['DL Online alpha=1e-4', 'DL Online alpha=3.16e-4', 
                  'DL Online alpha=1e-3', 'DL Online alpha=3.16e-3', 
                  'DL Online alpha=1e-2', 'DL Online alpha=3.16e-2', 
                  'DL Online alpha=1e-1', 'DL Online alpha=3.16e-1', 
                  'DL Online alpha=1e0']
    dl_sparses = ['Sparse alpha=1e-4', 'Sparse alpha=3.16e-4', 
                  'Sparse alpha=1e-3', 'Sparse alpha=3.16e-3', 
                  'Sparse alpha=1e-2', 'Sparse alpha=3.16e-2', 
                  'Sparse alpha=1e-1', 'Sparse alpha=3.16e-1', 
                  'Sparse alpha=1e0']
    # alpha_vals = np.logspace(-4, 0, 9)
    alpha_vals = [.0001, .000316, .001, .00316, .01, .0316, .1, .316, 1.]
    fname_tags = ['alpha00001', 'alpha0000316', 'alpha0001', 'alpha000316', 
                  'alpha001', 'alpha00316', 'alpha01', 'alpha0316', 'alpha1']
    
    # #####
    
    # dl_onlines = ['DL Online alpha=1e-1']
    # dl_sparses = ['Sparse alpha=1e-1']
    # alpha_vals = [.1]
    # fname_tags = ['alpha01']
    
    # ##################################################################

    # Load dataframe
    df0 = pd.read_csv(str(Path(
        eval_dir, eval_metric.replace('_', '-') + '_evaluation_' + dataset + 
        '_cvrun.tsv')), sep='\t')
    
    # Filter for training session
    df1 = df0[df0['Train Session'] == 1]

    # Filter columns that will be used to compute ...
    # ... Repeated-Measures (RM) average
    df1 = df1[[
        'Subject', 'HBP Group', 'HBP Group + Individual', 'HBP Individual', 
        'DR GICA-U', 'NNLS', 
        'DL Online alpha=1e-4', 'DL Online alpha=3.16e-4', 
        'DL Online alpha=1e-3', 'DL Online alpha=3.16e-3',
        'DL Online alpha=1e-2', 'DL Online alpha=3.16e-2',
        'DL Online alpha=1e-1', 'DL Online alpha=3.16e-1',
        'DL Online alpha=1e0', 
        'Sparse alpha=1e-4', 'Sparse alpha=3.16e-4', 
        'Sparse alpha=1e-3', 'Sparse alpha=3.16e-3',
        'Sparse alpha=1e-2', 'Sparse alpha=3.16e-2',
        'Sparse alpha=1e-1', 'Sparse alpha=3.16e-1',
        'Sparse alpha=1e0']]  

    for (dl_online, dl_sparse, alpha_val, fname_tag) in zip(
         dl_onlines, dl_sparses, alpha_vals, fname_tags):

        # Compute mean across RM
        df2 = df1.groupby(['Subject']).mean()

        # Get columns-of-interest
        df2 = df2[[
            'HBP Group', 'HBP Group + Individual', 'HBP Individual',
            'DR GICA-U', 'NNLS', dl_online, dl_sparse]]
        
        # Compute some paired t-tests
        hbp_group = df2["HBP Group"].values
        hbp_group_individual = df2["HBP Group + Individual"]
        hbp_individual = df2["HBP Individual"]

        dualreg = df2["DR GICA-U"]
        nnls = df2["NNLS"]

        odl = df2[dl_online]
        sparse = df2[dl_sparse]

        _, p_hbp_ggi = ttest_rel(hbp_group_individual, hbp_group)
        _, p_hbp_gii = ttest_rel(hbp_group_individual, hbp_individual)
        _, p_hbp_gi = ttest_rel(hbp_group, hbp_individual)    

        _, p_hbpi_dr = ttest_rel(hbp_individual, dualreg)
        _, p_dr_nnls = ttest_rel(dualreg, nnls)
        _, p_dualreg_odl = ttest_rel(dualreg, odl)

        _, p_hbpi_odl = ttest_rel(hbp_individual, odl)
        _, p_hbpi_sparse = ttest_rel(hbp_individual, sparse)   
        _, p_odl_sparse = ttest_rel(odl, sparse)

        _, p_nnls_sparse = ttest_rel(nnls, sparse)

        pvals = [p_hbp_ggi, p_hbp_gii, p_hbp_gi, p_hbpi_dr, p_dr_nnls, 
                 p_dualreg_odl, p_hbpi_odl, p_hbpi_sparse, p_odl_sparse]
        
        # Do the plots
        sci_alpha = "{:.2e}".format(alpha_val)

        plot_evaluation(df2, pvals, dl_online, dl_sparse, sci_alpha, str(Path(
            eval_dir, eval_metric.replace('_', '-') + '_evaluation_' + 
            dataset + '_tr-ses1_cvrun_' + fname_tag + '.png')), 
            evaluation=eval_metric)