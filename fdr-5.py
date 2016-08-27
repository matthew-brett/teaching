q = 0.05
plt.plot(i, p_values, 'b.', label='$p(i)$')
# [...]
plt.plot(i, q * i / N, 'r', label='$q i / N$')
# [...]
plt.xlabel('$i$')
# <...>
plt.ylabel('$p$')
# <...>
plt.legend()
# <...>
