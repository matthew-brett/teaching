N = n_time_points
M = n_hrf_points
shifted_scaled_hrfs = np.zeros((N, N + M - 1))
for i in range(N):
    input_value = neural_signal[i]
    # Storing the shifted, scaled HRF
    shifted_scaled_hrfs[i, i : i + n_hrf_points] = hrf_signal * input_value
bold_signal_again = np.sum(shifted_scaled_hrfs, axis=0)

# We check that the result is almost exactly the same
# (allowing for tiny differences due to the order of +, * operations)
import numpy.testing as npt
npt.assert_almost_equal(bold_signal, bold_signal_again)
