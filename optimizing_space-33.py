def fancy_xy_trans_slice(img_slice, x_y_trans):
    """ Make a copy of `img_slice` translated by `x_y_trans` voxels
# ...
    x_y_trans is a sequence or array length 2, containing
    the (x, y) translations in voxels.
# ...
    Values in `x_y_trans` can be positive or negative, and
    can be floats.
    """
    x_y_trans = np.array(x_y_trans)
    # Resample image using bilinear interpolation (order=1)
    trans_slice = snd.affine_transform(img_slice, [1, 1], -x_y_trans, order=1)
    return trans_slice
