def hrf(t):
    "A hemodynamic response function"
    return t ** 8.6 * np.exp(-t / 0.547)

hrf_times = np.arange(0, 20, 0.1)
hrf_signal = hrf(hrf_times)
plt.plot(hrf_times, hrf_signal)
# [...]
plt.xlabel('time (seconds)')
# <...>
plt.ylabel('BOLD signal')
# <...>
plt.title('Estimated BOLD signal for event at time 0')
# <...>
