n = symbols('n')
multi_assoc_noprior = dict(t_s = (1 - beta ** n),
                          t_ns = beta ** n,
                          f_s = 1 - (1 - alpha) ** n,
                          f_ns = (1 - alpha) ** n)
