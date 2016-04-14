times = np.arange(0, 40)  # One time point per second
n_time_points = len(times)
neural_signal = np.zeros(n_time_points)
neural_signal[4:7] = 1  # A 3 second event
neural_signal[10] = 1
neural_signal[20] = 3
hrf_times = np.arange(20)
hrf_signal = hrf(hrf_times)  # The HRF at one second time resolution
n_hrf_points = len(hrf_signal)
bold_signal = np.convolve(neural_signal, hrf_signal)
times_and_tail = np.arange(n_time_points + n_hrf_points - 1)
fig, axes = plt.subplots(3, 1, figsize=(8, 15))
axes[0].plot(times, neural_signal)
# [...]
axes[0].set_title('Neural signal, 1 second resolution')
# <...>
axes[1].plot(hrf_times, hrf_signal)
# [...]
axes[1].set_title('Hemodynamic impulse response, 1 second resolution')
# <...>
axes[2].plot(times_and_tail, bold_signal)
# [...]
axes[2].set_title('Predicted BOLD signal from convolution, 1 second resolution')
# <...>
