def cost_function(x_trans):
    # Function can use image slices defined in the global scope
    # Calculate X_t - image translated by x_trans
    unshifted = fancy_x_trans_slice(shifted_mid_vol1, x_trans)
    # Return mismatch measure for the translated image X_t
    return correl_mismatch(unshifted, mid_vol0)
