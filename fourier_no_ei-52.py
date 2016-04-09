# Reconstruct unshifted cos from dot product projection
c_unshifted = cos_x.dot(c_1) / c_1.dot(c_1)
proj_onto_c1 = c_unshifted * c_1
plt.plot(ns, proj_onto_c1)
plt.title('Reconstructed unshifted cosine')
