# First we make the shifted HRFs
shifted_hrfs = np.zeros((N, N + M - 1))
for i in range(N):
    # Storing the shifted HRF without scaling
    shifted_hrfs[i, i : i + n_hrf_points] = hrf_signal
# Then do the scaling
shifted_scaled_hrfs = np.zeros((N, N + M - 1))
for i in range(N):
    input_value = neural_signal[i]
    # Scaling the stored HRF by the input value
    shifted_scaled_hrfs[i, :] = shifted_hrfs[i, :] * input_value
# Then the sum
bold_signal_again = np.sum(shifted_scaled_hrfs, axis=0)
# This gives the same result, once again
npt.assert_almost_equal(bold_signal, bold_signal_again)
