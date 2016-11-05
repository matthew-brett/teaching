import scipy.stats
# Make a t distribution object for t with 20 degrees of freedom
t_dist = scipy.stats.t(20)
# Plot the PDF
t_values = np.linspace(-4, 4, 1000)
plt.plot(t_values, t_dist.pdf(t_values))
# [...]
plt.xlabel('t value')
# <...>
plt.ylabel('probability for t value')
# <...>
plt.title('PDF for t distribution with df=20')
# <...>
