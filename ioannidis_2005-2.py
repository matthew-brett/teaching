from sympy.abc import alpha, beta, pi # get symbolic variables
post_prob = (1 - beta) * pi / ((1 - beta) * pi + alpha * (1 - pi))
post_prob
# pi*(-beta + 1)/(alpha*(-pi + 1) + pi*(-beta + 1))
