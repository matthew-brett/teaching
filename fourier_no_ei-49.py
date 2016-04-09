# Reconstruct shifted cos from dot product projection
c_cos_shifted = cos_x_shifted.dot(c_1) / c_1.dot(c_1)
c_sin_shifted = cos_x_shifted.dot(s_1) / s_1.dot(s_1)
proj_onto_c1 = c_cos_shifted * c_1
proj_onto_s1 = c_sin_shifted * s_1
reconstructed = proj_onto_c1 + proj_onto_s1
plt.plot(ns, reconstructed)
plt.title('Reconstructed shifted cosine')
assert np.allclose(reconstructed, cos_x_shifted)
