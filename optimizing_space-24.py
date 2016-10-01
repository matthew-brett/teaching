fine_mismatches = []
fine_translations = np.linspace(-25, 15, 100)
for t in fine_translations:
    unshifted = fancy_x_trans_slice(shifted_mid_vol1, t)
    mismatch = correl_mismatch(unshifted, mid_vol0)
    fine_mismatches.append(mismatch)
