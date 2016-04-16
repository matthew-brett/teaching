def mean_abs_mismatch(slice0, slice1):
    """ Mean absoute difference between images
    """
    return np.mean(np.abs(slice0 - slice1))
