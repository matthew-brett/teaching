def x_trans_slice(img_slice, x_vox_trans):
    """ Make a new copy of `img_slice` translated by `x_vox_trans` voxels
# ...
    `x_vox_trans` can be positive or negative
    """
    # Make a 0-filled array of same shape as `img_slice`
    trans_slice = np.zeros(img_slice.shape)
    # Use slicing to select voxels out of the image and move them
    # Up or down on the first (x) axis
    if x_vox_trans < 0:
        trans_slice[:x_vox_trans, :] = img_slice[-x_vox_trans:, :]
    elif x_vox_trans == 0:
        trans_slice[:, :] = img_slice
    else:
        trans_slice[x_vox_trans:, :] = img_slice[:-x_vox_trans, :]
    return trans_slice
