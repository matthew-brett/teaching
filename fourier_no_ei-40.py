cos_x_shifted = 3 * np.cos(t_1 + 0.8)
plt.plot(t_1, cos_x_shifted)
# [...]
print('Dot product of shifted cosine with c_1', cos_x_shifted.dot(c_1))
# Dot product of shifted cosine with c_1 33.4419220487
