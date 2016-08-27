neural_signal[i_time_4] = 2  # An impulse with amplitude 2
plt.plot(times, neural_signal)
# [...]
plt.xlabel('time (seconds)')
# <...>
plt.ylabel('neural signal')
# <...>
plt.ylim(0, 2.2)
# (...)
plt.title('Neural model for amplitude 2 impulse')
# <...>
