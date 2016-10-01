R = pi / (1 - pi)
ppv = (1 - beta) * R / (R - beta * R + alpha)
# Is this the same as our formula above?
simplify(ppv - post_prob) == 0
# True
