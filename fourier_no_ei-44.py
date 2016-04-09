ns = np.arange(N)
t_1 = 2 * np.pi * ns / N
plt.plot(ns, np.cos(t_1), 'o:')
plt.plot(ns, np.sin(t_1), 'o:')
plt.xlim(0, N-1)
plt.xlabel('n')
