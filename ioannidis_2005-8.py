# Same as Ioannidis formulation?
# This from Ioannidies 2005:
ppv_bias = (
    ((1 - beta) * R + u * beta * R) /
    (R + alpha - beta * R + u - u * alpha + u * beta * R)
   )
# Is this the same as our formula above?
simplify(ppv_bias - post_prob_bias) == 0
# True
