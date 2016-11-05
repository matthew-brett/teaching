# Plot the CDF
plt.plot(t_values, t_dist.cdf(t_values))
# [...]
plt.xlabel('t value')
# <...>
plt.ylabel('probability for t value <= t')
# <...>
plt.title('CDF for t distribution with df=20')
# <...>
