vec_r_2 = vec_r_1 * 2
vec_c_2 = np.cos(vec_r_2)
vec_s_2 = np.sin(vec_r_2)
plt.plot(vec_n, vec_c_2, 'o:', label=r'$\vec{c^2}$')
# [...]
plt.plot(vec_n, vec_s_2, 'x:', label=r'$\vec{s^2}$')
# [...]
plt.xlabel('Vector index $n$')
# <...>
plt.ylabel('$c^2_n$')
# <...>
plt.legend()
# <...>
