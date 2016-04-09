t_1 = 2 * np.pi * ns / N
cos_x = 3 * np.cos(t_1)
c_1 = np.cos(t_1)
X = np.fft.fft(cos_x)
print('First DFT coefficient for single cosine', X[1])
print('Dot product of single cosine with c_1', cos_x.dot(c_1))
print('3 * dot product of c_1 with itself', 3 * c_1.T.dot(c_1))
