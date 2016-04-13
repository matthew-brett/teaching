import numpy as np
import numpy.linalg as npl
# Make some random, but predictable data
np.random.seed(1966)
X = np.random.multivariate_normal([0, 0], [[3, 1.5], [1.5, 1]], size=50).T
X.shape
# (2, 50)
