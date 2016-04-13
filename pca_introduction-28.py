remaining = X - projected
plt.scatter(remaining[0], remaining[1], label='remaining')
# <...>
plt.arrow(0, 0, u_best[0], u_best[1], width=0.01, color='r')
# <...>
plt.annotate('$\hat{u_{best}}$', u_best, xytext=(20, 20), textcoords='offset points', fontsize=20)
# <...>
plt.legend(loc='upper left')
# <...>
plt.axis('equal')
# (...)
