multi_bias_assoc_noprior = dict(
   t_s = (1 - beta ** n) + u * beta ** n,
   t_ns = beta ** n - u * beta ** n,
   f_s = 1 - (1 - alpha) ** n + u * (1 - alpha) ** n,
   f_ns = (1 - alpha) ** n - u * (1 - alpha)**n)
