# Formula from Ioannidies 2005:
ppv_multi = R * (1 - beta ** n) / (R + 1 - (1 - alpha) ** n - R * beta ** n)
# Is this the same as our formula above?
simplify(ppv_multi - post_prob_multi) == 0
# True
