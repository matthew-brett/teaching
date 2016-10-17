#: Standard imports
import numpy as np
# Make numpy print 4 significant digits for prettiness
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
# Default to nearest neighbor interpolation, gray colormap
import matplotlib
matplotlib.rcParams['image.interpolation'] = 'nearest'
matplotlib.rcParams['image.cmap'] = 'gray'
