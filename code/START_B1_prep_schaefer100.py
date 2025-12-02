import nitools as nt
import os
import numpy as np
from py_util_dx.py_utils import setProjectPath


projectPath, mainResultsPath = setProjectPath()
surface_helpers_dir = os.path.join(projectPath, 'surface_helpers')

lid, cmap, names = nt.read_lut(os.path.join(surface_helpers_dir, f'atl-schaefer100_pre-cleaning.lut'))
cmap = cmap[np.arange(1, len(cmap), 2)] / 255
names = list(lid[np.arange(0, len(lid), 2)])
nums = list(lid[np.arange(1, len(lid), 2)])

pass
