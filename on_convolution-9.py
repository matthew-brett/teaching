neural_signal[i_time_4:i_time_4 + 5] = 2
plt.plot(times, neural_signal)
# [...]
plt.xlabel('time (seconds)')
# <...>
plt.ylabel('neural signal')
# <...>
plt.ylim(0, 3.2)
# (...)
plt.title('Neural model including event lasting 0.5 seconds')
# <...>
