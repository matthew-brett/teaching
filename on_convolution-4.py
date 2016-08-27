n_hrf_points = len(hrf_signal)
bold_signal = np.zeros(n_time_points)
bold_signal[i_time_4:i_time_4 + n_hrf_points] = hrf_signal
plt.plot(times, bold_signal)
# [...]
plt.xlabel('time (seconds)')
# <...>
plt.ylabel('bold signal')
# <...>
plt.title('Output BOLD signal for event at time=4')
# <...>
