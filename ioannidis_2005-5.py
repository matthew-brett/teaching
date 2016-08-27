u = symbols('u')
bias_assoc_noprior = dict(t_s = (1 - beta) + u * beta,
                          t_ns = beta - u * beta,
                          f_s = alpha + u * (1 - alpha),
                          f_ns = (1 - alpha) - u * (1 - alpha))
