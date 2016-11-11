def x_trans_slice(img_slice, x_vox_trans):
    """ Return copy of `img_slice` translated by `x_vox_trans` voxels
# ...
    Parameters
    ----------
    img_slice : array shape (M, N)
        2D image to transform with translation `x_vox_trans`
    x_vox_trans : int
        Number of pixels (voxels) to translate `img_slice`; can be
        positive or negative.
# ...
    Returns
    -------
    img_slice_transformed : array shape (M, N)
        2D image translated by `x_vox_trans` pixels (voxels).
    """
    # Make a 0-filled array of same shape as `img_slice`
    trans_slice = np.zeros(img_slice.shape)
    # Use slicing to select voxels out of the image and move them
    # up or down on the first (x) axis
    if x_vox_trans < 0:
        trans_slice[:x_vox_trans, :] = img_slice[-x_vox_trans:, :]
    elif x_vox_trans == 0:
        trans_slice[:, :] = img_slice
    else:
        trans_slice[x_vox_trans:, :] = img_slice[:-x_vox_trans, :]
    return trans_slice
