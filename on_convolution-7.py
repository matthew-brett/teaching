neural_signal[i_time_4] = 2  # An impulse with amplitude 2
i_time_10 = np.where(times == 10)[0]  # index of value 10 in "times"
neural_signal[i_time_10] = 1  # An impulse with amplitude 1
i_time_20 = np.where(times == 20)[0]  # index of value 20 in "times"
neural_signal[i_time_20] = 3  # An impulse with amplitude 3
plt.plot(times, neural_signal)
# [...]
plt.xlabel('time (seconds)')
# <...>
plt.ylabel('neural signal')
# <...>
plt.ylim(0, 3.2)
# (...)
plt.title('Neural model for three impulses')
# <...>
