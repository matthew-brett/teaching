#################################
Notes on the Bonferroni threshold
#################################

The Bonferroni threshold is a family-wise error threshold. That is, it
treats a set of tests as one *family*, and the threshold is designed to
control the probability of detecting *any* positive tests in the family
(set) of tests, if the null hypothesis is true.

*****************
Family-wise error
*****************

The Bonferroni correction uses a result from probability theory to
estimate the probability of finding *any* p value below a threshold
:math:`\theta`, given a set (family) of :math:`n` p values.

When we have found a threshold :math:`\theta` that gives a probability
:math:`\le \alpha` that *any* p value will be :math:`\lt \theta`, then
the threshold :math:`\theta` can be said to control the *family-wise
error rate* at level :math:`\alpha`.

*****************************
Not the Bonferroni correction
*****************************

The inequality used for the Bonferroni is harder to explain than a
simpler but related correction, called the Šidák correction.

We will start with that, and then move on to the Bonferroni correction.

The probability that all :math:`n` tests are *above* p value threshold
:math:`\theta`, *assuming tests are independent*:

.. math::

   (1 - \theta)^n

Chance that one or more p values are :math:`\le \theta`:

.. math::

   1 - (1 - \theta)^n

We want a uncorrected p value threshold :math:`\theta` such that the
expression above equals some desired family-wise error (FWE) rate
:math:`\alpha_{fwe}`. For example we might want a p value threshold
:math:`\theta` such that there is probability (:math:`\alpha_{fwe}`) of
0.05 that there is one or more test with :math:`p \le \theta` in a
family of :math:`n` tests, on the null hypothesis:

.. math::

   \alpha_{fwe} = 1 - (1 - \theta)^n

Solve for :math:`\theta`:

.. math::

   \theta = 1 - (1 - \alpha_{fwe})^{1 / n}

So, if we have 10 tests, and we want the threshold :math:`\theta` to
control :math:`\alpha_{fwe}` at $0.05$:

.. nbplot::

    >>> def sidak_thresh(alpha_fwe, n):
    ...     return 1 - (1 - alpha_fwe)**(1./n)
    ...
    >>> print(sidak_thresh(0.05, 10))
    0.00511619689182

*************************
The Bonferroni correction
*************************

:math:`\newcommand{\P}{\mathbb P}` The Bonferroni correction uses a
result from probability theory, called Boole's inequality. The result is
by George Boole, of *boolean* fame. Boole's inequality applies to the
situation where we have a set of events $A_1, A_2, A_3, \ldots $, each
with some probability of occurring ${P}(A_1), {P}(A_2), {P}(A_3) \ldots
$. The inequality states that the probability of one or more of these
events occurring is no greater than the sum of the probabilities of the
individual events:

.. math::

   \P\biggl(\bigcup_{i} A_i\biggr) \le \sum_i {\mathbb P}(A_i).

You can read the :math:`\cup` symbol here as "or" or "union".
:math:`\P\biggl(\bigcup_{i} A_i\biggr)` is the probability of the
*union* of all events, and therefore the probability of one or more
event occurring.

Boole's inequality is true because:

.. math::

   \P(A \cup B) = P(A) + P(B) - P(A \cap B)

where you can read :math:`\cap` as "and" or "intersection". Because
:math:`P(A \cap B) \ge 0`:

.. math::

   \P(A \cup B) \le P(A) + P(B)

In our case we have :math:`n` tests (the family of tests). Each test
that we label as significant is an event. Therefore the sum of the
probabilities of all possible events is :math:`n\theta`.
:math:`{\mathbb P}\biggl(\bigcup_{i} A_i\biggr)` is our probability of
family-wise error :math:`\alpha_{fwe}`. To get a threshold
:math:`\theta` that controls family-wise error at :math:`\alpha`, we
need:

.. math::

   \frac{\alpha_{fwe}}{n} \le \theta

For :math:`n=10` tests and an :math:`\alpha_{fwe}` of 0.05:

.. nbplot::

    >>> def bonferroni_thresh(alpha_fwe, n):
    ...     return alpha_fwe / n
    ...
    >>> print(bonferroni_thresh(0.05, 10))
    0.005

The Bonferroni correction does not assume the tests are independent.

As we have seen, Boole's inequality relies on:

.. math::

   \P(A \cup B) = P(A) + P(B) - P(A \cap B) \implies \\
   \P(A \cup B) \le P(A) + P(B)

This means that the Bonferroni correction will be conservative (the
threshold will be too low) when the tests are positively dependent
(:math:`P(A \cap B) \gg 0`).

The Bonferroni
:math:`\theta_{Bonferroni} = \alpha_{fwe} \space / \space n` is always
smaller (more conservative) than the Šidák correction
:math:`\theta_{Šidák}` for :math:`n \ge 1`, but it is close:

.. nbplot::

    >>> import numpy as np
    >>> n_tests = np.arange(1, 11)  # n = 1 through 10
    >>> # The exact threshold for independent p values
    >>> print(sidak_thresh(0.05, n_tests))
    [ 0.05    0.0253  0.017   0.0127  0.0102  0.0085  0.0073  0.0064  0.0057
      0.0051]

.. nbplot::

    >>> # The Bonferroni threshold for the same alpha, n
    >>> print(bonferroni_thresh(0.05, n_tests))
    [ 0.05    0.025   0.0167  0.0125  0.01    0.0083  0.0071  0.0063  0.0056
      0.005 ]
