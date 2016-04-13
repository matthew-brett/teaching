def line_projection(u, X):
    """ Return columns of X projected onto line defined by u
    """
    u = u.reshape(1, 2)  # A row vector
    c_values = u.dot(X)  # c values for scaling u
    projected = u.T.dot(c_values)
    return projected
