import os
import numpy as np


def setProjectPath():
    """
    setProjectPath specifies the main directory of the project and the results folder

    usage
    projectPath, mainResultsPath = setProjectPath()

    last modified: 2023.11.06
    """
    pythonAnalysisPath = os.getcwd()
    projectPath = os.path.join(pythonAnalysisPath, '..')    # exploration path
    mainResultsPath = os.path.join(projectPath, 'results')

    return projectPath, mainResultsPath


def sqmat2vec(sqmat, upperORlower='upper'):
    """
    extracts the upper or lower triangular part of a square matrix and return as a vector

    """

    if upperORlower == 'upper':
        coords = np.where(np.triu(np.ones(sqmat.shape), k=1) > 0)
    elif upperORlower == 'lower':
        coords = np.where(np.tril(np.ones(sqmat.shape), k=1) > 0)
    else:
        coords = np.nan
        exit(-1)
    vec = sqmat[coords]
    return vec


def r2z(r):
    z = np.log(np.divide(1+r, 1-r)) / 2
    return z


def z2r(z):
    r = np.divide(np.exp(2*z)-1, np.exp(2*z)+1)
    return r


def fisherMean(vec):
    z = r2z(vec)
    z_mean = np.nanmean(z)
    r_mean = z2r(z_mean)
    return r_mean

