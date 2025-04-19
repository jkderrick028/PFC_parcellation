"""
Reshape DU15NET Probabilistic Atlas and
resample from fsaverage6 (41k) to fsLR_32K

author: Ana Luisa Pinho
email: agrilopi@uwo.ca

created: November 23, 2024
last update: December, 2024

Compatibility: Python 3.9.20
"""

import os
import glob
import gzip
import shutil
import numpy as np
import nibabel as nib

from nibabel.cifti2 import (
    Cifti2Matrix,
    Cifti2MatrixIndicesMap,
    Cifti2BrainModel,
    Cifti2NamedMap,
    Cifti2Header,
    Cifti2Image,
    Cifti2VertexIndices
)
from nibabel.gifti import GiftiImage, GiftiDataArray

from neuromaps import transforms


# ###################### FUNCTIONS #####################################

def one_hot_encode_with_zeros(arr, num_parcels):
    """
    One-hot encode an array with zeros

    Parameters:
    -----------
    arr: np.ndarray (1, n_vertices)
        Array with parcellation assignment for each vertex
    num_parcels: int    
        Number of parcels in the parcellation           
    """

    n = arr.shape[1]
    arr_flat = arr.flatten()
    one_hot = np.zeros((num_parcels, n), dtype=int)
    nonzero_indices = np.nonzero(arr_flat)[0]
    nonzero_values = arr_flat[nonzero_indices]
    one_hot[nonzero_values - 1, nonzero_indices] = 1
    
    return one_hot


def create_cifti(num_vertices, data, output_path):
    """
    Create a CIFTI-2 file of the group probabilities in a pre-specified 
    surface space from number of vertices

    Parameters:
    -----------
    data: np.ndarray (n_parcels, n_vertices)
        Array with the group probabilities
    output_path: str

    Returns:
    --------
    cifti_img: nib.Cifti2Image
        CIFTI-2 image with the group probabilities in the pre-specified 
        surface space
    """

    # Create brain models for the spatial dimension
    bm_left = Cifti2BrainModel(
        index_offset=0,
        index_count=num_vertices,
        model_type='CIFTI_MODEL_TYPE_SURFACE',
        brain_structure='CIFTI_STRUCTURE_CORTEX_LEFT',
        vertex_indices=Cifti2VertexIndices(np.arange(num_vertices, 
                                                     dtype=np.uint32))
    )

    bm_right = Cifti2BrainModel(
        index_offset=num_vertices,
        index_count=num_vertices,
        model_type='CIFTI_MODEL_TYPE_SURFACE',
        brain_structure='CIFTI_STRUCTURE_CORTEX_RIGHT',
        vertex_indices=Cifti2VertexIndices(np.arange(num_vertices, 
                                                     dtype=np.uint32))
    )

    # # Create Scalar Mappings for the First Dimension
    # Create a list of named maps for the scalars
    scalar_maps = []
    for i in range(n_parcels):
        map_name = f'Parcel {i + 1}'
        named_map = Cifti2NamedMap(map_name)
        scalar_maps.append(named_map)

    # # Construct the CIFTI-2 Header
    # Create the Matrix Indices Map for the scalar dimension 
    # (dimension index 0)
    mim_scalar = Cifti2MatrixIndicesMap(
        applies_to_matrix_dimension=[0],
        indices_map_to_data_type='CIFTI_INDEX_TYPE_SCALARS',
        maps=scalar_maps
    )
    # Create the Matrix Indices Map for the spatial dimension 
    # (dimension index 1)
    mim_space = Cifti2MatrixIndicesMap(
        applies_to_matrix_dimension=[1],
        indices_map_to_data_type='CIFTI_INDEX_TYPE_BRAIN_MODELS',
        maps=[bm_left, bm_right]
    )

    # Create the CIFTI-2 Matrix and append both mappings
    matrix = Cifti2Matrix()
    matrix.append(mim_scalar)  # First dimension mapping
    matrix.append(mim_space)   # Second dimension mapping

    # Retrieve scalar axis and BrainModelAxis from the matrix
    scalar_axis = matrix.get_axis(0)
    brain_model_axis = matrix.get_axis(1)

    # Modify nvertices in BrainModelAxis
    modified_nvertices = {
        "CIFTI_STRUCTURE_CORTEX_LEFT": num_vertices,
        "CIFTI_STRUCTURE_CORTEX_RIGHT": num_vertices
    }
    brain_model_axis.nvertices.update(modified_nvertices)

    # Print modification
    print("Modified BrainModelAxis nvertices:")
    print(brain_model_axis.nvertices)

    # Create an header based on the CIFTI-2 axes with the modified 
    # BrainModelAxis
    header = Cifti2Header.from_axes((scalar_axis, brain_model_axis))

    # Create the CIFTI-2 image
    cifti_img = Cifti2Image(dataobj=data, header=header)

    # Save the CIFTI-2 file
    nib.save(cifti_img, output_path)

    return cifti_img


def extract_hemi_data(data, indices_list):
    """
    Extract data for each hemisphere

    Parameters:
    -----------
    data: np.ndarray (n_parcels, n_vertices)
        Array with the group probabilities
    indices_list: list
        List with the indices of each hemisphere

    Returns:
    --------
    hemi_data: np.ndarray (n_parcels, n_vertices)
        Array with the group probabilities for each hemisphere 
    """

    hemi_data = []
    for start, end in indices_list:
        hemi_data.append(data[:, start:end])

    return np.concatenate(hemi_data, axis=1)


def gifti_conversion(data_array):
    """
    Save data as an array of GIFTI images

    Parameters:
    -----------
    data_array: np.ndarray (n_parcels, n_vertices)
        Array with the data to be saved
    file_path: str
        Path to save the GIFTI file

    Returns:
    --------
    gifti_array: np.ndarray (n_parcels,)
        Array of GIFTI images
    """

    num_parcels = data_array.shape[0]

    gii_list = []
    for i in range(num_parcels):  # For each map
        gifti_img = GiftiImage()
        darray = GiftiDataArray(data=data_array[i, :])
        gifti_img.add_gifti_data_array(darray)
        gii_list.append(gifti_img)
        del gifti_img

    return gii_list


def resample(gii_list, hemi):
    """
    Resample GIFTI images from fsaverage6 to fsLR_32k

    Parameters:
    -----------
    gii_list: list
        List of GIFTI images to be resampled
    hemi: str
        Hemisphere ('L' or 'R')

    Returns:
    --------
    gii_fslr32k_list: list
        List of resampled GIFTI images
    """

    gii_fslr32k_list = []
    for gii in gii_list:  # For each map
        # Resample from fsaverage6 to fsaverage
        gii_fsaverage = transforms.fsaverage_to_fsaverage(
            gii, target_density='164k', hemi=hemi)       
        # Resample from fsaverage to fsLR_32k
        gii_fslr32k = transforms.fsaverage_to_fslr(gii_fsaverage, hemi=hemi)
        # Append to the list
        gii_fslr32k_list.append(gii_fslr32k[0])

    return gii_fslr32k_list


# ####################### INPUTS #######################################

# Define some paths
work_dir = os.path.dirname(os.path.abspath(__file__))
atlases_dir = os.path.join(work_dir, 'atlases')
data_pardir = os.path.join(os.path.expanduser('~'), 'Dropbox/Diedrichsenlab')
fsaverage6_dir = os.path.join(data_pardir, 'fsaverage6')
fsaverage6_atlas_dir = os.path.join(fsaverage6_dir, 
                                    'atl-MSHBM_Prior_15_fsaverage6')

# Path of original DU15NET group probabilistic atlas in fsLR_32k
du15net_agreeprob_path = os.path.join(
    atlases_dir, 'DU15NET_AgreeProb_fsLR_32k.dscalar.nii')

# Paths of individual parcellations in fsaverage6
du15net_iparcels_folder = os.path.join(fsaverage6_atlas_dir, 'dscalar_15')
du15net_iparcels_paths = glob.glob(os.path.join(
    du15net_iparcels_folder, '*.nii'))
du15net_iparcels_arrs = np.array([
    nib.load(du15net_iparcels_path).get_fdata() 
    for du15net_iparcels_path in du15net_iparcels_paths])

# Output paths
gprob_fsaverage6_path = os.path.join(
    atlases_dir, 'DU15NET_GroupProb_fsaverage6.dscalar.nii')
gprob_fslr32_path = os.path.join(
    atlases_dir, 'DU15NET_GroupProb_fsLR_32k.dscalar.nii')

n_subjects = du15net_iparcels_arrs.shape[0]
n_parcels = np.unique(du15net_iparcels_arrs[0])[1:].shape[0]
n_vertices = du15net_iparcels_arrs.shape[2]

NUM_VERTICES_FSAVERAGE6 = 40962
NUM_VERTICES_FSLR32K = 32492

# ######################## RUN #########################################

if __name__ == "__main__":

    # Pre-allocate the group array with zeros
    one_hot_grouparr = np.zeros((n_subjects, n_parcels, n_vertices))

    # Iterate over each individual parcellation
    for s, du15net_iparcels_arr in enumerate(du15net_iparcels_arrs):
        one_hot_arr = one_hot_encode_with_zeros(
            du15net_iparcels_arr.astype(int), n_parcels)
        one_hot_grouparr[s] = one_hot_arr
        del one_hot_arr

    # Compute group probabilities
    gprob_fsaverage6_data = np.mean(one_hot_grouparr, axis=0).astype(
        np.float32)
    
    # Create CIFTI file in fsaverage6
    gprob_cifti_fsaverage6 = create_cifti(NUM_VERTICES_FSAVERAGE6, 
                                          gprob_fsaverage6_data, 
                                          gprob_fsaverage6_path)

    # Get the brain models from the header
    header = gprob_cifti_fsaverage6.header
    bm_map = header.get_index_map(1)  # Dimension 1 is the spatial dimension
    brain_models = bm_map.brain_models

    # # Separate Data for Each Hemisphere
    # Initialize lists to hold data and vertex indices
    lh_indices = []
    rh_indices = []
    # Collect index ranges for each hemisphere
    for bm in brain_models:
        if bm.brain_structure == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            lh_indices.append((
                bm.index_offset, bm.index_offset + bm.index_count))
        elif bm.brain_structure == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            rh_indices.append((
                bm.index_offset, bm.index_offset + bm.index_count))

    # Extract data for left and right hemispheres -- Shape: (15, 40962)
    lh_data = extract_hemi_data(gprob_fsaverage6_data, lh_indices)  
    rh_data = extract_hemi_data(gprob_fsaverage6_data, rh_indices)

    # Store data separately from each hemisphere as a GIFTI image
    lh_fsaverage6_list = gifti_conversion(lh_data)
    rh_fsaverage6_list = gifti_conversion(rh_data)

    # Resample from fsaverage6 to fsLR_32k
    lh_fslr32k_list = resample(lh_fsaverage6_list, 'L')
    rh_fslr32k_list = resample(rh_fsaverage6_list, 'R')

    # Extract data from GIFTI files
    lh_gprob_fslr32_data = np.array([lh_fslr32k.darrays[0].data 
                                     for lh_fslr32k in lh_fslr32k_list])
    rh_gprob_fslr32_data = np.array([rh_fslr32k.darrays[0].data 
                                     for rh_fslr32k in rh_fslr32k_list])
    
    # Concatenate data from both hemispheres
    gprob_fslr32_data = np.concatenate(
        [lh_gprob_fslr32_data, rh_gprob_fslr32_data], axis=1).astype(
            np.float32)

    # Create CIFTI file in fsLR_32k
    create_cifti(NUM_VERTICES_FSLR32K, gprob_fslr32_data, gprob_fslr32_path)