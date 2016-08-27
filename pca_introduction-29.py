u_best_orth = np.array([np.cos(angle_best + np.pi / 2), np.sin(angle_best + np.pi / 2)])
plt.scatter(remaining[0], remaining[1], label='remaining')
# <...>
plt.arrow(0, 0, u_best[0], u_best[1], width=0.01, color='r')
# <...>
plt.arrow(0, 0, u_best_orth[0], u_best_orth[1], width=0.01, color='g')
# <...>
plt.annotate('$\hat{u_{best}}$', u_best, xytext=(20, 20), textcoords='offset points', fontsize=20)
# <...>
plt.annotate('$\hat{u_{orth}}$', u_best_orth, xytext=(20, 20), textcoords='offset points', fontsize=20)
# <...>
plt.axis('equal')
# (...)
