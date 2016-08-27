def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    pxy = hgram / float(np.sum(hgram)) # Convert bins to probability
    px = np.sum(pxy, 1) # marginal for x over y
    py = np.sum(pxy, 0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
