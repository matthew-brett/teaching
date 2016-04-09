s_1 = np.sin(t_1)
plt.plot(t_1, cos_x, label='3 * cos wave')
# [...]
plt.plot(t_1, cos_x_shifted, label='3 * cos wave, shifted')
# [...]
plt.legend()
# <...>
print('Dot product of unshifted cosine with c_1', cos_x.dot(c_1))
# Dot product of unshifted cosine with c_1 48.0
print('Dot product of unshifted cosine with s_1', cos_x.dot(s_1))
# Dot product of unshifted cosine with s_1 -8.53851287343e-16
print('Dot product of shifted cosine with c_1', cos_x_shifted.dot(c_1))
# Dot product of shifted cosine with c_1 33.4419220487
print('Dot product of shifted cosine with s_1', cos_x_shifted.dot(s_1))
# Dot product of shifted cosine with s_1 -34.4330923632
