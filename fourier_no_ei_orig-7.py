fig, axes = plt.subplots(6, 1, figsize=(8, 5))
ns = np.arange(N)
one_cycle = 2 * np.pi * ns / N
for k in range(6):
    t_k = k * one_cycle
    axes[k].plot(ns, np.cos(t_k), label='cos')
    axes[k].plot(ns, np.sin(t_k), label='sin')
    axes[k].set_xlim(0, N-1)
    axes[k].set_ylim(-1.1, 1.1)
# [...)
axes[0].legend()
# <...>
plt.tight_layout()
