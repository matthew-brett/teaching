first_i = i[:30]
plt.plot(first_i, mixed_p_values[:30], 'b.', label='$p(i)$')
# [...]
plt.plot(first_i, q * first_i / N, 'r', label='$q i / N$')
# [...]
plt.xlabel('$i$')
# <...>
plt.ylabel('$p$')
# <...>
plt.legend()
# <...>
