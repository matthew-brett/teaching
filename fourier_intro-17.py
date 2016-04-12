beta = 1.1
vec_v = np.cos(vec_r_1 + beta)
plt.plot(vec_n, vec_c_1, 'o:', label='Unshifted cos')
# [...]
plt.plot(vec_n, vec_v, 'x:', label='Shifted cos')
# [...]
plt.legend()
# <...>
