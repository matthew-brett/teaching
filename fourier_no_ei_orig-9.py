# Recalculate the forward transform with C and S
X_again = C.dot(x) - 1j * S.dot(x)
assert np.allclose(X, X_again)  # same result as for np.fft.fft
# Recalculate the inverse transform
x_again = 1. / N * C.dot(X) + 1j / N * S.dot(X)
assert np.allclose(x, x_again)  # as for np.fft.ifft, we get x back
