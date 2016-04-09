t_1 = 2 * np.pi * ns / N
cos_x = 3 * np.cos(t_1)
c_1 = np.cos(t_1)
X = np.fft.fft(cos_x)
print('First DFT coefficient for single cosine', X[1])
# First DFT coefficient for single cosine (48-3.89264601266e-14j)
print('Dot product of single cosine with c_1', cos_x.dot(c_1))
# Dot product of single cosine with c_1 48.0
print('3 * dot product of c_1 with itself', 3 * c_1.T.dot(c_1))
# 3 * dot product of c_1 with itself 48.0
