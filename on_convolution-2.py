neural_signal = np.zeros(n_time_points)
i_time_4 = np.where(times == 4)[0]  # index of value 4 in "times"
neural_signal[i_time_4] = 1  # A single spike at time == 4
plt.plot(times, neural_signal)
# [...]
plt.xlabel('time (seconds)')
# <...>
plt.ylabel('neural signal')
# <...>
plt.ylim(0, 1.2)
# (...)
plt.title("Neural model for very brief event at time 4")
# <...>
