bold_signal = np.zeros(n_time_points)
for i in range(5):
    bold_signal[i_time_4 + i:i_time_4  + i + n_hrf_points] += hrf_signal * 2
bold_signal[i_time_10:i_time_10 + n_hrf_points] += hrf_signal * 1
bold_signal[i_time_20:i_time_20 + n_hrf_points] += hrf_signal * 3
plt.plot(times, bold_signal)
# [...]
plt.xlabel('time (seconds)')
# <...>
plt.ylabel('bold signal')
# <...>
plt.title('Output BOLD signal with event lasting 0.5 seconds')
# <...>
