mismatches = []
translations = range(-25, 15)  # Candidate values for t
for t in translations:
    # Make the translated image Y_t
    unshifted = x_trans_slice(shifted_mid_vol1, t)
    # Calculate the mismatch
    mismatch = mean_abs_mismatch(unshifted, mid_vol0)
    # Store it for later
    mismatches.append(mismatch)
