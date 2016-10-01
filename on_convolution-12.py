bold_signal = np.convolve(neural_signal, hrf_signal)
plt.plot(times_and_tail, bold_signal)
# [...]
plt.xlabel('time (seconds)')
# <...>
plt.ylabel('bold signal')
# <...>
plt.title('Our algorithm is the same as convolution')
# <...>
