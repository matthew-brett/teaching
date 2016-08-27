i = 25
bold_i = neural_vector.dot(shifted_hrfs[:, i])

npt.assert_almost_equal(bold_i, bold_signal[i])
