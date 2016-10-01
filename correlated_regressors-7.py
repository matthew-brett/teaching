noise = np.random.normal(size=times.shape)
Y = signal + noise
plt.plot(times, signal)
# [...]
plt.plot(times, Y, '+')
# [...]
