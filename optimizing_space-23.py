def fancy_x_trans_slice(img_slice, x_vox_trans):
    """ Return copy of `img_slice` translated by `x_vox_trans` voxels
# ...
    Parameters
    ----------
    img_slice : array shape (M, N)
        2D image to transform with translation `x_vox_trans`
    x_vox_trans : float
        Number of pixels (voxels) to translate `img_slice`; can be
        positive or negative, and does not need to be integer value.
    """
    # Resample image using bilinear interpolation (order=1)
    trans_slice = snd.affine_transform(img_slice, [1, 1], [-x_vox_trans, 0], order=1)
    return trans_slice
