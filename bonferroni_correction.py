# $\newcommand{L}[1]{\| #1 \|}\newcommand{VL}[1]{\L{ \vec{#1} }}\newcommand{R}[1]{\operatorname{Re}\,(#1)}\newcommand{I}[1]{\operatorname{Im}\, (#1)}$
#
# ## Notes on the Bonferroni threshold
#
# The Bonferroni threshold is a family-wise error threshold. That is, it
# treats a set of tests as one *family*, and the threshold is designed to
# control the probability of detecting *any* positive tests in the family
# (set) of tests, if the null hypothesis is true.
#
# ### Family-wise error
#
# The Bonferroni correction uses a result from probability theory to
# estimate the probability of finding *any* p value below a threshold
# $\theta$, given a set (family) of $n$ p values.
#
# When we have found a threshold $\theta$ that gives a probability
# $\le \alpha$ that *any* p value will be $\lt \theta$, then
# the threshold $\theta$ can be said to control the *family-wise
# error rate* at level $\alpha$.
#
# ### Not the Bonferroni correction
#
# The inequality used for the Bonferroni is harder to explain than a
# simpler but related correction, called the Šidák correction.
#
# We will start with that, and then move on to the Bonferroni correction.
#
# The probability that all $n$ tests are *above* p value threshold
# $\theta$, *assuming tests are independent*:
#
# $$
# (1 - \theta)^n
# $$
#
# Chance that one or more p values are $\le \theta$:
#
# $$
# 1 - (1 - \theta)^n
# $$
#
# We want a uncorrected p value threshold $\theta$ such that the
# expression above equals some desired family-wise error (FWE) rate
# $\alpha_{fwe}$. For example we might want a p value threshold
# $\theta$ such that there is probability ($\alpha_{fwe}$) of
# 0.05 that there is one or more test with $p \le \theta$ in a
# family of $n$ tests, on the null hypothesis:
#
# $$
# \alpha_{fwe} = 1 - (1 - \theta)^n
# $$
#
# Solve for $\theta$:
#
# $$
# \theta = 1 - (1 - \alpha_{fwe})^{1 / n}
# $$
#
# So, if we have 10 tests, and we want the threshold $\theta$ to
# control $\alpha_{fwe}$ at $0.05$:

def sidak_thresh(alpha_fwe, n):
    return 1 - (1 - alpha_fwe)**(1./n)

print(sidak_thresh(0.05, 10))

# # The Bonferroni correction
#
# $\newcommand{\P}{\mathbb P}$ The Bonferroni correction uses a
# result from probability theory, called Boole’s inequality. The result is
# by George Boole, of *boolean* fame. Boole’s inequality applies to the
# situation where we have a set of events $A_1, A_2, A_3, \ldots $, each
# with some probability of occurring ${P}(A_1), {P}(A_2), {P}(A_3) \ldots
# $. The inequality states that the probability of one or more of these
# events occurring is no greater than the sum of the probabilities of the
# individual events:
#
# $$
# \P\biggl(\bigcup_{i} A_i\biggr) \le \sum_i {\mathbb P}(A_i).
# $$
#
# You can read the $\cup$ symbol here as “or” or “union”.
# $\P\biggl(\bigcup_{i} A_i\biggr)$ is the probability of the
# *union* of all events, and therefore the probability of one or more
# event occurring.
#
# Boole’s inequality is true because:
#
# $$
# \P(A \cup B) = P(A) + P(B) - P(A \cap B)
# $$
#
# where you can read $\cap$ as “and” or “intersection”. Because
# $P(A \cap B) \ge 0$:
#
# $$
# \P(A \cup B) \le P(A) + P(B)
# $$
#
# In our case we have $n$ tests (the family of tests). Each test
# that we label as significant is an event. Therefore the sum of the
# probabilities of all possible events is $n\theta$.
# ${\mathbb P}\biggl(\bigcup_{i} A_i\biggr)$ is our probability of
# family-wise error $\alpha_{fwe}$. To get a threshold
# $\theta$ that controls family-wise error at $\alpha$, we
# need:
#
# $$
# \frac{\alpha_{fwe}}{n} \le \theta
# $$
#
# For $n=10$ tests and an $\alpha_{fwe}$ of 0.05:

def bonferroni_thresh(alpha_fwe, n):
    return alpha_fwe / n

print(bonferroni_thresh(0.05, 10))

# The Bonferroni correction does not assume the tests are independent.
#
# As we have seen, Boole’s inequality relies on:
#
# $$
# \P(A \cup B) = P(A) + P(B) - P(A \cap B) \implies \\
# \P(A \cup B) \le P(A) + P(B)
# $$
#
# This means that the Bonferroni correction will be conservative (the
# threshold will be too low) when the tests are positively dependent
# ($P(A \cap B) \gg 0$).
#
# The Bonferroni
# $\theta_{Bonferroni} = \alpha_{fwe} \space / \space n$ is always
# smaller (more conservative) than the Šidák correction
# $\theta_{Šidák}$ for $n \ge 1$, but it is close:

import numpy as np
np.set_printoptions(precision=4)  # print to 4 decimal places
n_tests = np.arange(1, 11)  # n = 1 through 10
# The exact threshold for independent p values
print(sidak_thresh(0.05, n_tests))

# The Bonferroni threshold for the same alpha, n
print(bonferroni_thresh(0.05, n_tests))
