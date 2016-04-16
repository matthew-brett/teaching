def fancy_cost_at_xy(x_y_trans):
    """ Give cost function at xy translation values `x_y_trans`
    """
    unshifted = fancy_xy_trans_slice(shifted_mid_vol1, x_y_trans)
    return correl_mismatch(unshifted, mid_vol0)
