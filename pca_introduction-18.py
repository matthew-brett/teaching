def line_remaining(u, X):
    """ Return vectors remaining after removing cols of X projected onto u
    """
    projected = line_projection(u, X)
    remaining = X - projected
    return remaining
