correl_mismatches = []
translations = range(-60, 50)  # Candidate values for t
for t in translations:
    unshifted = x_trans_slice(shifted_mid_vol1, t)
    mismatch = correl_mismatch(unshifted, mid_vol0)
    correl_mismatches.append(mismatch)
