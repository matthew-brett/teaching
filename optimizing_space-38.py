def cost_at_xyz(x_y_z_trans):
    """ Give cost function value at xyz translation values `x_y_z_trans`
    """
    unshifted = xyz_trans_vol(shifted_vol1, x_y_z_trans)
    return correl_mismatch(unshifted, vol0)
