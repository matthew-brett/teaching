def xyz_trans_vol(vol, x_y_z_trans):
    """ Make a new copy of `vol` translated by `x_y_z_trans` voxels
# ...
    x_y_z_trans is a sequence or array length 3, containing
    the (x, y, z) translations in voxels.
# ...
    Values in `x_y_z_trans` can be positive or negative,
    and can be floats.
    """
    x_y_z_trans = np.array(x_y_z_trans)
    # [1, 1, 1] says to do no zooming or rotation
    # Resample image using trilinear interpolation (order=1)
    trans_vol = snd.affine_transform(vol, [1, 1, 1], -x_y_z_trans, order=1)
    return trans_vol
