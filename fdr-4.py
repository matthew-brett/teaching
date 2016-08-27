p_values = np.sort(p_values)
i = np.arange(1, N+1) # the 1-based i index of the p values, as in p(i)
plt.plot(i, p_values, '.')
# [...]
plt.xlabel('$i$')
# <...>
plt.ylabel('p value')
# <...>
