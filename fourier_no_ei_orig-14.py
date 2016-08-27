complex_x = np.array(  # A Random array of complex numbers
      [ 0.61-0.83j, -0.82-0.12j, -0.50+1.14j,  2.37+1.67j,  1.62+0.69j,
        1.61-0.06j,  0.54-0.73j,  0.89-1.j  ,  0.17-0.71j,  0.75-0.01j,
       -1.06-0.14j, -2.53-0.33j,  1.74+0.83j,  1.34-0.64j,  1.47+0.71j,
        0.82+0.4j , -1.59-0.58j,  0.13-1.02j,  0.47-0.73j,  1.45+1.31j,
        1.32-0.28j,  1.58-2.13j,  0.75-0.43j,  1.24+0.4j ,  0.02+1.08j,
        0.07-0.57j, -1.21+1.08j,  1.38+0.54j, -1.35+0.3j , -0.61+1.08j,
       -0.96+1.81j, -1.95+1.64j])
complex_X = np.fft.fft(complex_x)  # Canned DFT
complex_X_again = C.dot(complex_x) - 1j * S.dot(complex_x)  # Our DFT
# We get the same result as the canned DFT
assert np.allclose(complex_X, complex_X_again)
