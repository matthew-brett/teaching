vec_c_1 = np.cos(vec_r_1)
vec_s_1 = np.sin(vec_r_1)
plt.plot(vec_n, vec_c_1, 'o:', label=r'$\vec{c^1}$')
# [...]
plt.plot(vec_n, vec_s_1, 'x:', label=r'$\vec{s^1}$')
# [...]
plt.xlabel('Vector index $n$')
# <...>
plt.ylabel('$c^1_n$')
# <...>
plt.legend()
# <...>
