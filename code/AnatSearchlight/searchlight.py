"""
Searchlight familiy class for creating searchlight structures and running searchlight analyses
within the defined structures.

created on 2024-05-16

Author: Diedrichsenlab
"""

import numpy as np
import pandas as pd
import nibabel as nb
import nitools as nt
import h5py # For IO with HDF5 files
from os import PathLike
import AnatSearchlight.pymvpa_surf as nno  # Nick Oosterhof's library for dijkstra distance

def load(fname, single=False, idx=None):
    """Loads a searchlight definition from an HDF5 file

    Args:
        fname (str): Filename
    Returns:
        Searchlight: Searchlight object
    """
    with h5py.File(fname, 'r') as hf:
        classname = hf.get('classname').asstr()[()]
        structure = hf.get('structure').asstr()[()]
        if classname == 'SearchlightVolume':
            S = SearchlightVolume(structure)
        elif classname == 'SearchlightSurface':
            S = SearchlightSurface(structure)
        else:
            raise ValueError(f"Classname {classname} not recognized")
        if single:
            S.load_single(hf, idx)
        else:
            S.load(hf)
        hf.close()
    return S


class Searchlight:
    """Base class for searchlight analyses. This class implements the basic behaviors of a searchlight.

    Attributes:
        classname (str): Name of the class
        structure (str): Structure name for the cifti file (e.g. 'cerebellum', 'left_hem', etc.)
        affine (ndarray): Affine transformation matrix of data and output space
        shape (tuple): Shape of the data and output space
        center_indx (ndarray): Voxel / vertex indices of the searchlight centers
        n_cent (int): Number of searchlight centers
        maxradius (float): Maximum distance from each searchlight center
        maxvoxels (int): Maximum number of voxels in the searchlight
        voxel_indx (ndarray): 3 x P array of voxel indices in functional space (i,j,k)
        voxlist (list): List of voxel numbers (index into voxel_indx) for each searchlight center
        voxmin (ndarray): Minimum voxel indices for each searchlight center
        voxmax (ndarray): Maximum voxel indices for each searchlight center
        radius (ndarray): Effective radius for each searchlight center
        nvoxels (ndarray): Number of voxels in the searchlight
    """

    def define(self):
        """ Computes the voxel list for a searchlight. Needs to be implemented by the child class."""
        pass

    def load(self,hf):
        """Loads all obligatory fields from the h5 file

        Args:
            hf (h5py.file): h5 file object (opened in read mode)
        """
        self.affine= np.array(hf.get('affine')) # Affine matrix for functional space
        self.shape = np.array(hf.get('shape'))  # shape of functional image
        self.center_indx = np.array(hf.get('center_indx')) # Center indices (voxel or vertex)
        self.voxel_indx = np.array(hf.get('voxel_indx')) # voxel indices of candidate voxels
        self.maxdist= np.array(hf.get('maxdist'))  # Maximum distance for each searchlight center
        self.maxradius = np.max(self.maxdist)  # Maximum radius of the searchlight
        self.voxlist = []  # List of indices into voxel_indx
        for i in range(len(hf.get('voxlist'))):
            self.voxlist.append(np.array(hf.get(f'voxlist/voxlist_{i}')))
        self.voxmin = np.array(hf.get('voxmin'))
        self.structure = hf.get('structure').asstr()[()]
        self.voxmax = np.array(hf.get('voxmax'))
        self.radius = np.array(hf.get('radius'))
        self.nvoxels = np.array(hf.get('nvoxels'))
        if self.center_indx.ndim == 1:
            self.n_cent = self.center_indx.shape[0]
        else:
            self.n_cent = self.center_indx.shape[1]
        if self.classname == 'SearchlightSurface':
            self.n_vertices = np.array(hf.get('n_vertices')) # Affine matrix for functional space

    def load_single(self, hf, idx):
        """Loads a single searchlight (idx) from HDF5 file"""
        self.affine = np.array(hf.get('affine'))  # Affine matrix for functional space
        self.shape = np.array(hf.get('shape'))  # shape of functional image
        self.voxel_indx = np.array(hf.get('voxel_indx'))  # voxel indices of candidate voxels
        self.voxlist = np.array(np.array(hf.get(f'voxlist/voxlist_{idx}')))

    def save(self,fname):
        """Saves the defined searchlight definition to hd5 file
        Args:
            fname (str): Filename
        """
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('classname',data = self.classname)
            hf.create_dataset('structure', data=self.structure)
            hf.create_dataset('affine', data=self.affine)
            hf.create_dataset('shape', data=self.shape)
            hf.create_dataset('center_indx', data=self.center_indx)
            hf.create_dataset('maxradius', data=self.maxradius)
            hf.create_dataset('maxvoxels', data=self.maxvoxels)
            hf.create_dataset('voxel_indx', data=self.voxel_indx)
            grp = hf.create_group('voxlist')
            for i in range(len(self.voxlist)):
                grp.create_dataset(f'voxlist_{i}', data=self.voxlist[i])
            hf.create_dataset('voxmin', data=self.voxmin)
            hf.create_dataset('voxmax', data=self.voxmax)
            hf.create_dataset('radius', data=self.radius)
            hf.create_dataset('nvoxels', data=self.nvoxels)
            if self.classname == 'SearchlightSurface':
                hf.create_dataset('n_vertices', data=self.n_vertices)
            hf.close()

    def run(self,inputfiles,mvpa_function,function_args={},verbose=True):
        """ Conducts a searchlight analysis for all the searchlights defined in vox_list.

        Args:
            inputfiles (list):
                List of filenames
            mvpa_fumction (fcn):
                Function that takes input data (N x n_voxels) and returns a scalar or vector as result.
                Function should be defined as:
                def mvpa_function(data, **kwargs):
            function_args (dictionary):
                Additional arguments to be passed to the mvpa_function as keyword arguments.
        Returns:
            results (ndarray):
                Either one-dimensional or two-dimensional ndarray. First dimension is
                the number of centers.
        """

        # Load the header from all the input files
        input_vols = [nb.load(f) for f in inputfiles]

        # Sample only the voxels that we require
        data = np.empty((len(input_vols),self.voxel_indx.shape[1]))
        if verbose:
            print(f"Loading data ")
        for i,v in enumerate(input_vols):
            if verbose:
                print(f"Volume number {i+1}")
            D = v.get_fdata()
            data[i,:] = D[self.voxel_indx[0],self.voxel_indx[1],self.voxel_indx[2]]

        # Call the mvpa function
        results = []
        for i in range(self.n_cent):
            if (np.mod(i,1000)==0) and verbose:
                print(f"Calculating searchlight {i} of {self.n_cent}")
            local_data = data[:,self.voxlist[i]]
            results.append(mvpa_function(local_data,**function_args))
        results = np.array(results)
        return results

    def run_parallel(self, inputfiles, mvpa_function, function_args={}, nargout=1, verbose=True, n_jobs=8):
        """
        Run searchlight parallel not serial. 
        """
        # load data with float32 to be more efficient (float 32 instead of 64)
        if isinstance(inputfiles, PathLike):
            inputfiles = nb.load(inputfiles)
        if isinstance(inputfiles, nb.Cifti2Image):
            inputfiles = nt.volume_from_cifti(inputfiles, struct_names=self.structure)
        if isinstance(inputfiles, nb.Nifti2Image):
            vol = inputfiles.get_fdata(dtype=np.float32, caching="unchanged")
            data = vol[self.voxel_indx[0], self.voxel_indx[1], self.voxel_indx[2]].T
        if isinstance(inputfiles, list):
            input_vols = [nb.load(f) for f in inputfiles]
            if verbose:
                print(f"Loading {len(input_vols)} volumes...")
            data = np.empty((len(input_vols), self.voxel_indx.shape[1]), dtype=np.float32)
            for i, v in enumerate(input_vols):
                # caching=unchanged prevents nibabel from keeping a copies in the memory 
                data[i, :] = v.get_fdata(dtype=np.float32, caching="unchanged")[
                                        self.voxel_indx[0], self.voxel_indx[1], self.voxel_indx[2]]

        # Prepare voxel lists to not send the whole self object to each worker
        voxlists = self.voxlist 

        # define the function for the workers
        def _process_one(i):
            v_idx = voxlists[i]
            if len(v_idx) == 0:
                return np.nan if nargout==1 else np.full(nargout, np.nan, dtype=float)
            # joblib seemingly memmaps 'data' automatically in the backend
            return mvpa_function(data[:, v_idx], **function_args)

        if verbose:
            print(f"Running searchlight on {self.n_cent} centers...")

        # Run in parallel
        with Parallel(n_jobs=n_jobs, batch_size=100, verbose=10) as parallel:
            results = parallel(delayed(_process_one)(i) for i in range(self.n_cent))

        return np.array(results)


class SearchlightVolume(Searchlight):
    """ Anatomically informed searchlights for 3d volumes, given an ROI image.
    Voxels we picked from the mask_img, if an extra mask_image is provided.
    """
    def __init__(self,structure='none'):
        """Constructor for SearchlightVolume with either fixed radius or fixed number of voxels.
        If both a set to a value, nvoxels is used up to a maximum of the radius.

        Args:
            structure (str):
                structure name for the cifti file (e.g. 'cerebellum', 'left_hem', etc.).
        """
        self.classname ='SearchlightVolume'
        self.structure = structure

    def define(self,roi_img,mask_img=None,maxradius=5,maxvoxels=np.inf,verbose=True):
        """ Calculates the voxel_list for a Volume-based searchlight .

        Args:
            roi_img (filename or NiftiImage):
                ROI binary mask image to define searchlight centers. Same space as the later input data.
            mask_img (filename or NiftiImage):
                Mask image to define input space (voxels to be used in the searchlight),
                By default (None) the ROI image is used here as well. Should be in the same space as the later input data.
            maxradius (float):
                Maximum searchlight radius - set to np.inf if you only want a fixed number of voxels
            maxvoxels (float):
                Max number of voxels in the searchlight - set to np.inf if you want a fixed radius
            verbose (bool):
                If True, print progress messages.
        """
        if isinstance(roi_img,str):
            self.roi_img = nb.load(roi_img)
        elif isinstance(roi_img,nb.Nifti1Image):
            self.roi_img = roi_img
        else:
            raise ValueError("roi_img must be a filename or Nifti1Image")

        if mask_img is not None:
            if isinstance(mask_img,str):
                self.mask_img = nb.load(mask_img)
            elif isinstance(mask_img,nb.Nifti1Image):
                self.mask_img = mask_img
            else:
                raise ValueError("mask_img must be a filename or Nifti1Image")
        else:
            self.mask_img = None

        # record the other parameters
        self.maxradius = maxradius
        self.maxvoxels = maxvoxels
        self.affine = self.roi_img.affine # Affine transformation matrix of data and output space
        self.shape = self.roi_img.shape   # Shape of the data and output space

        # Define the center indices
        i,j,k = np.where(self.roi_img.get_fdata())
        self.center_indx = np.array([i,j,k]).astype('int16') # Index of each center
        self.n_cent = len(i) # number of centers
        center_coords = nt.affine_transform_mat(self.center_indx,self.affine) # coordinates of each center

        # If a mask image is provided, we use the voxels from the mask image
        if self.mask_img is not None:
            i,j,k = np.where(self.mask_img.get_fdata())
            self.voxel_indx = np.array([i,j,k]).astype('int16')
            voxel_coords = nt.affine_transform_mat(self.voxel_indx,self.affine)
        else:
            self.voxel_indx = self.center_indx
            voxel_coords = center_coords

        self.voxlist = []
        self.voxmin = np.zeros((self.n_cent,3),dtype='int16') # Bottom left voxel
        self.voxmax = np.zeros((self.n_cent,3),dtype='int16')
        self.maxdist = np.zeros(self.n_cent)
        self.radius = np.zeros(self.n_cent)
        self.nvoxels = np.zeros(self.n_cent)

        for i in range(self.n_cent):
            if (np.mod(i,1000)==0) and verbose:
                print(f"Processing center {i} of {self.n_cent}")
            dist = nt.euclidean_dist_sq(center_coords[:,i],voxel_coords).squeeze()
            vn = np.where(dist<(self.maxradius**2))[0]
            if len(vn)>self.maxvoxels:
                ind=np.argsort(dist[vn])[:self.maxvoxels]
                vn = vn[ind]
            self.voxlist.append(vn.astype('uint32'))
            self.voxmin[i,:] = np.min(self.voxel_indx[:,vn],axis=1)
            self.voxmax[i,:] = np.max(self.voxel_indx[:,vn],axis=1)
            self.radius[i] = np.sqrt(np.max(dist[vn]))
            self.nvoxels[i] = len(vn)  # Number of voxels in the searchlight

    def data_to_nifti(self,results,outfilename = None):
        """ Returns as nifti file with the results of the searchlight analysis.
        Args:
            results (np.array):
                Results of the searchlight analysis
                Can either be one-dimensional (n_centers,) -> 3d nifti
                Or two-dimensional (n_centers,n_results) -> 4d nifti
            outfilename (str):
                Filename to save the nifti image to. If None, the image is returned, but not saved.
        Returns
            img (nb.Nifti1Image):
                Nifti image with the results of the searchlight analysis.
        """
        if results.shape[0] != self.n_cent:
            raise ValueError(f"Results must have the same number of elements as the number of searchlights ({self.n_cent})")

        # Put the data into a 4d-matrix
        if (results.ndim ==1):
            result_size=1
        else:
            result_size = results.shape[1]
        if result_size==1:
            img_shape = self.shape
        else:
            img_shape = np.append(self.shape,result_size)

        results_img = np.zeros(img_shape,dtype='float32')*np.nan
        results_img[self.center_indx[0,:],self.center_indx[1,:],self.center_indx[2,:]] = results

        # First deal with a single 4d-nifti as an output
        img = nb.Nifti1Image(results_img,self.affine)
        if outfilename is not None:
            nb.save(img,outfilename)
        return img

class SearchlightSurface(Searchlight):
    """ Surface-based searchlights for a single hemisphere, given a individual surface.
    Voxels we picked from the mask_img, for each vertex on the surface.
    """

    def __init__(self,structure ='none',depths=[0,0.2,0.4,0.6,0.8,1]):
        """Constructor for SearchlightVolume with either fixed radius or fixed number of voxels.
        If both a set to a value, nvoxels is used up to a maximum of the radius.

        Args:
            structure (str):
                structure name for the cifti file (e.g. 'left_hem', 'right_hem').
        """
        self.classname ='SearchlightSurface'
        self.structure = structure
        self.depths = np.array(depths)  # Depth for sampling
        self.n_points = len(self.depths)  # Number of points to sample



    def define(self,surfs,mask_img,roi=None,maxradius=10,maxvoxels=np.inf,verbose=True):
        """ Calculates the voxel_list for a Volume-based searchlight .

        Args:
            surfs (list of strs):
                List of surface gifti file names used to define the searchlight centers.
            mask_img (filename or NiftiImage):
                Mask image to define input space (voxels to be used in the searchlight)
            roi (filename, GiftiImage, ndrarray):
                Define searchlight centers. If not given, the searchlight centers are calculated for all vertices.
            radius (float):
                Maximum searchlight radius - set to a large value if you only want a fixed number of voxels
            nvoxels (int):
                Number of voxels in the searchlight. set , the searchlight will be defined by a constant radius.
            verbose (bool):
                If True, print progress messages.
        """
        # Get the two surfaces
        if not isinstance(surfs,list):
            raise ValueError("surfs must be a list")
        self.surfs = surfs
        surfaces = [nb.load(f) for f in surfs]

        # Check that the surfaces are compatible
        if len(surfaces) != 2:
            raise ValueError("surfs must contain exactly two surfaces")
        if not all([isinstance(s,nb.gifti.GiftiImage) for s in surfaces]):
            raise ValueError("Surfaces must be GiftiImage objects")
        c1 = surfaces[0].darrays[0].data
        c2 = surfaces[1].darrays[0].data
        f1 = surfaces[0].darrays[1].data
        if c2.shape[0] != c1.shape[0]:
            raise ValueError("Surfaces must have the same number of vertices")

        # Get the mask image
        if isinstance(mask_img,str):
            self.mask_img = nb.load(mask_img)
        elif isinstance(mask_img,nb.Nifti1Image):
            self.mask_img = mask_img
        else:
            raise ValueError("roi_img must be a filename or Nifti1Image")

        # Record definition parameters:
        self.maxradius = maxradius
        self.maxvoxels = maxvoxels
        self.affine = self.mask_img.affine # Affine transformation matrix of data and output space
        self.shape = self.mask_img.shape   # Shape of the data and output space
        self.n_vertices = c1.shape[0]  # Number of vertices in the surface

        # Use the roi defition
        if (roi is None):
            self.center_indx = np.arange(self.n_vertices).astype('int16') # Index of each center
        elif isinstance(roi,np.ndarray):
            if roi.ndim == 1:
                self.center_indx = roi.astype('int16')
            else:
                raise ValueError("roi must be a 1d array")
        elif isinstance(roi,str):
            roi= nb.load(roi)
            if isinstance(roi,nb.gifti.GiftiImage):
                self.center_indx = np.where(roi.darrays[0].data>0)[0].astype('int16')
            else:
                raise ValueError("roi image must be a GiftiImage ")
        else:
            raise ValueError("roi must be a filename of GiftiImage or ndarray")
        self.n_cent = self.center_indx.shape[0] # number of centers

        # Get the indices for all the points being sampled
        # We get the unique voxels between the two surfaces (voxel_indx)
        # And the n_points x n_vertices numbers of voxels (voxels)
        lin_indices = np.zeros((self.n_points,self.n_vertices),dtype=int)
        for i in range(self.n_points):
            c = (1-self.depths[i])*c1.T+self.depths[i]*c2.T
            linvoxind,good = nt.coords_to_linvidxs(c,self.mask_img,mask=True)
            linvoxind[~good] = -1  # Set invalid voxels to -1
            lin_indices[i] = linvoxind.T
        flat_lindx = lin_indices.flatten()
        flat_lindx = flat_lindx[flat_lindx>-1]
        unique_lindices = np.unique(flat_lindx)  # Remove duplicates
        self.voxel_indx = np.stack(np.unravel_index(unique_lindices, self.mask_img.shape))
        # Compute mid-depth surface
        mid_depth = (c1 + c2) / 2
        midsurf = nno.Surface(mid_depth, f1)

        # Define the voxel list for each searchlight center
        self.voxlist = []  # List of indices into voxel_indx
        self.voxmin = np.zeros((self.n_cent,3),dtype='int16') # Bottom left voxel
        self.voxmax = np.zeros((self.n_cent,3),dtype='int16')
        self.radius = np.zeros(self.n_cent)
        self.nvoxels = np.zeros(self.n_cent)

        for i in range(self.n_cent):
            dist_dict = midsurf.dijkstra_distance(self.center_indx[i],self.maxradius)
            can_nodes = np.array([int(k) for k,v in dist_dict.items()])
            can_dist = np.array([np.double(v) for k,v in dist_dict.items()])
            can_lindicies = lin_indices[:,can_nodes].T.flatten()  # These are the voxels numbers sorted  by distance
            can_vox_dist = np.tile(can_dist,(self.n_points,1)).T.flatten()  # Distances to the voxels
            _,vorder = np.unique(can_lindicies,return_index=True)  # Remove duplicates and return index of closest occurrance
            vorder = np.sort(vorder)
            can_linin_sorted = can_lindicies[vorder]  # Sorted voxels distance
            can_voxdist_sorted = can_vox_dist[vorder]  # Sorted distances to the voxels (along the surface)
            # Remove voxels that are not in the mask
            goodv = can_linin_sorted>-1
            can_linin_sorted = can_linin_sorted[goodv]
            can_voxdist_sorted = can_voxdist_sorted[goodv]
            # >>> if there are no voxels in searchlight
            if can_linin_sorted.size == 0:
                print(f"No voxels for center {i} of {self.n_cent}")
                self.voxlist.append(np.array([], dtype='uint32'))
                self.voxmin[i, :] = np.array([-1, -1, -1]) #np.array([np.nan, np.nan, np.nan])
                self.voxmax[i, :] = np.array([-1, -1, -1]) #np.array([np.nan, np.nan, np.nan])
                self.radius[i] = np.nan
                self.nvoxels[i] = np.nan   # set to nan or mean count will be biased
                continue
            # <<<
            if np.isinf(maxvoxels): # take all the voxels within the radius
                vi=can_linin_sorted
                maxdist = can_voxdist_sorted[-1]
            else:
                vi = can_linin_sorted[:maxvoxels]
                maxdist = can_voxdist_sorted[len(vi)-1]

            # Get the voxel numbers in the unique_lindices
            vn = np.array([np.where(unique_lindices==v)[0][0] for v in vi],dtype ='uint32')
            self.voxlist.append(vn)
            self.voxmin[i,:] = np.min(self.voxel_indx[:,vn],axis=1)
            self.voxmax[i,:] = np.max(self.voxel_indx[:,vn],axis=1)
            self.radius[i] = maxdist  # Maximum distance in the searchlight
            self.nvoxels[i] = len(vn)  # Number of voxels in the searchlight

            if (np.mod(i,1000)==0) and verbose:
                print(f"Processing center {i} of {self.n_cent}, size={maxdist}, n voxels={len(vn)}")

    def data_to_cifti(self,data,outfilename = None,row_names=None):
        """ Returns a CIFTI file with the results of the searchlight analysis.
        Args:
            results (np.array):
                Results of the searchlight analysis (ncenters,n_results).
                Can either be one-dimensional or two-dimensional
            """
        # check if the results are in the right size
        if data.ndim == 1:
            data = data.reshape((-1,1))
        if data.shape[0] != self.n_cent:
            raise ValueError(f"Results must have the same number of elements as the number of searchlights ({self.n_cent})")

        # Use vertex mask to create a BrainModelAxis
        vertex_mask = np.zeros((self.n_vertices,),dtype='bool')
        vertex_mask[self.center_indx] = True
        bm = nb.cifti2.BrainModelAxis.from_mask(vertex_mask, name=self.structure)

        # Make the ScalarAxis
        if row_names is None:
            row_names = [f"row {r:03}" for r in range(data.shape[1])]
        row_axis = nb.cifti2.ScalarAxis(row_names)

        header = nb.Cifti2Header.from_axes((row_axis, bm))
        cifti_img = nb.Cifti2Image(dataobj=data.T, header=header)

        # Save if requested
        if outfilename is not None:
            nb.save(cifti_img, outfilename)
        return cifti_img


class SearchlightSet():
    def __init__(self,list_of_searchlights):
        pass
