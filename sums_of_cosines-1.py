from __future__ import print_function, division

import numpy as np

def predicted_cos_sum(a, d, N):
    d2 = d / 2.
    if np.allclose(np.sin(d2), 0):
        return N * np.cos(a)
    return np.sin(N * d2) / np.sin(d2) * np.cos(a + (N - 1) * d2)
# ...
def predicted_sin_sum(a, d, N):
    d2 = d / 2.
    if np.allclose(np.sin(d2), 0):
        return N * np.sin(a)
    return np.sin(N * d2) / np.sin(d2) * np.sin(a + (N - 1) * d2)
# ...
def actual_cos_sum(a, d, N):
    angles = np.arange(N) * d + a
    return np.sum(np.cos(angles))
# ...
def actual_sin_sum(a, d, N):
    angles = np.arange(N) * d + a
    return np.sum(np.sin(angles))
