def as_row_vector(v):
    " Convert 1D vector to row vector "
    return v.reshape((1, -1))

neural_vector = as_row_vector(neural_signal)
# The scaling and summing by the magic of matrix multiplication
bold_signal_again = neural_vector.dot(shifted_hrfs)
# This gives the same result as previously, yet one more time
npt.assert_almost_equal(as_row_vector(bold_signal), bold_signal_again)
