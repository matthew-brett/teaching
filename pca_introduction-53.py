# Divide out reconstructed S values
S_as_column = S_from_C.reshape((2, 1))
np.allclose(C / S_as_column, VT)
# True
