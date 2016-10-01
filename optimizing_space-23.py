def fancy_x_trans_slice(img_slice, x_vox_trans):
    """ Make a new copy of `img_slice` translated by `x_vox_trans` voxels
# ...
    `x_vox_trans` can be positive or negative, and can be a float.
    """
    # Resample image using bilinear interpolation (order=1)
    trans_slice = snd.affine_transform(img_slice, [1, 1], [-x_vox_trans, 0], order=1)
    return trans_slice
