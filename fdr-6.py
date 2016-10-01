N_signal = 20
N_noise = N - N_signal
noise_z_values = np.random.normal(size=N_noise)
# Add some signal with very low z scores / p values
signal_z_values = np.random.normal(loc=-2.5, size=N_signal)
mixed_z_values = np.sort(np.concatenate((noise_z_values, signal_z_values)))
mixed_p_values = normal_distribution.cdf(mixed_z_values)
plt.plot(i, mixed_p_values, 'b.', label='$p(i)$')
# [...]
plt.plot(i, q * i / N, 'r', label='$q i / N$')
# [...]
plt.xlabel('$i$')
# <...>
plt.ylabel('$p$')
# <...>
plt.legend()
# <...>
