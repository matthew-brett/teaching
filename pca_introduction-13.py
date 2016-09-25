u_guessed_row = u_guessed.reshape((1, 2))  # A row vector
c_values = u_guessed_row.dot(X)  # c values for scaling u
# scale u by values to get projection
projected = u_guessed_row.T.dot(c_values)
