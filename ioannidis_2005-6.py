bias_assoc = bias_assoc_noprior.copy()
bias_assoc['t_s'] *= pi
bias_assoc['t_ns'] *= pi
bias_assoc['f_s'] *= 1 - pi
bias_assoc['f_ns'] *= 1 - pi
