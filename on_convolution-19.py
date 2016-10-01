bold_signal_again = shifted_hrfs.T.dot(neural_vector.T)
# Exactly the same, but transposed
npt.assert_almost_equal(as_row_vector(bold_signal), bold_signal_again.T)
