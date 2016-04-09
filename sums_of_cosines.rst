
Sum of sines and cosines
------------------------

* Samuel Greitzer "Many cheerful Facts" Arbelos 4 (1986), no. 5, 14-17.
* Michael P. Knapp `Sines and cosines of angles in arithmetic progression
  <http://evergreen.loyola.edu/mpknapp/www/papers/knapp-sv.pdf>_` Mathematics
  Magazine 82.5 (2009): 371-372.

The proof on this page follows Samuel Greitzner's "Many cheerful facts".

Theorems
~~~~~~~~

For a positive integer :math:`N` and real numbers :math:`a, d`:

.. math::

   R \triangleq \frac{\sin(N \frac{1}{2}d)}{\sin(\frac{1}{2} d)} \\
   \sum_{n=0}^{N-1} \cos(a + nd) =
   \begin{cases}
   N \cos a & \text{if } \sin(\frac{1}{2}d) = 0 \\
   R \cos ( a + (N - 1) \frac{1}{2} d) & \text{otherwise}
   \end{cases}
   \\
   \sum_{n=0}^{N-1} \sin(a + nd) =
   \begin{cases}
   N \sin a & \text{if } \sin(\frac{1}{2}d) = 0 \\
   R \sin ( a + (N - 1) \frac{1}{2} d) & \text{otherwise}
   \end{cases}

Numerical check
---------------

Is this numerically the case?

.. nbplot::

    >>> from __future__ import print_function, division

    >>> import numpy as np

    >>> def predicted_cos_sum(a, d, N):
    ...     d2 = d / 2.
    ...     if np.allclose(np.sin(d2), 0):
    ...         return N * np.cos(a)
    ...     return np.sin(N * d2) / np.sin(d2) * np.cos(a + (N - 1) * d2)
    ...
    >>> def predicted_sin_sum(a, d, N):
    ...     d2 = d / 2.
    ...     if np.allclose(np.sin(d2), 0):
    ...         return N * np.sin(a)
    ...     return np.sin(N * d2) / np.sin(d2) * np.sin(a + (N - 1) * d2)
    ...
    >>> def actual_cos_sum(a, d, N):
    ...     angles = np.arange(N) * d + a
    ...     return np.sum(np.cos(angles))
    ...
    >>> def actual_sin_sum(a, d, N):
    ...     angles = np.arange(N) * d + a
    ...     return np.sum(np.sin(angles))

.. nbplot::

    >>> # When sin(d / 2) != 0
    >>> print('cos',
    ...       predicted_cos_sum(4, 0.2, 17),
    ...       actual_cos_sum(4, 0.2, 17))
    cos 7.7038472261 7.7038472261
    >>> print('sin',
    ...       predicted_sin_sum(4, 0.2, 17),
    ...       actual_sin_sum(4, 0.2, 17))
    sin -6.27049470825 -6.27049470825

.. nbplot::

    >>> # When sin(d / 2) ~ 0
    >>> print('cos : sin(d/2) ~ 0;',
    ...       predicted_cos_sum(4, np.pi * 2, 17),
    ...       actual_cos_sum(4, np.pi * 2, 17))
    cos : sin(d/2) ~ 0; -11.1119415547 -11.1119415547
    >>> print('sin : sin(d/2) ~ 0;',
    ...       predicted_sin_sum(4, np.pi * 2, 17),
    ...       actual_sin_sum(4, np.pi * 2, 17))
    sin : sin(d/2) ~ 0; -12.8656424202 -12.8656424202

Proof
-----

The basic order of play is to rearrange the sum so that the terms in the
current iteration of the sum cancel terms in the previous iteration, and
we can therefore get rid of the sum. This is a `telescoping
series <https://en.wikipedia.org/wiki/Telescoping_series>`__.

We will do the cosine series first. The sine proof is almost identical.

Cosine sum
~~~~~~~~~~

For reasons that will become clear later, we start with the case where
:math:`\sin(\frac{1}{2} d) = 0`.

If :math:`\sin(\frac{1}{2} d) = 0` then :math:`d` is a multiple of
:math:`2 \pi` and:

.. math::


   \cos(a) = \cos(a + d) = \cos(a + 2 d) ...

and:

.. math::


   \sum_{n=0}^{N-1} \cos(a + nd) = N \cos a

Now we cover the case where :math:`\sin(\frac{1}{2} d) \ne 0`.

From the `angle sum rules <./angle_sum.html>`__:

.. math::


   \sin(\alpha + \beta) = \sin \alpha \cos \beta + \cos \alpha \sin \beta \\
   \sin(\alpha - \beta) = \sin \alpha \cos \beta - \cos \alpha \sin \beta \\
   \implies \\
   \sin(\alpha + \beta) - \sin(\alpha - \beta) = 2\cos \alpha \sin \beta

Let:

.. math::


   C \triangleq \sum_{n=0}^{N-1}\cos(a + nd)

Only if :math:`\sin(\frac{1}{2} d) \ne 0`, we can safely multiply both
sides by :math:`2 \sin(\frac{1}{2} d)`:

.. math::


   2 \sin(\frac{1}{2} d) C = \sum_{n=0}^{N-1}2 \cos(a + nd) \sin(\frac{1}{2}d)

Now we use the angle sum derivation above. Let :math:`\alpha = a + nd`,
:math:`\beta = \frac{1}{2} d`:

.. math::


   2 \sin(\frac{1}{2} d) C =
   \sum_{n=0}^{N-1} \bigg ( \sin(a + (n + \frac{1}{2}) d) - \sin(a + (n -
   \frac{1}{2}) d) \bigg )

Writing out the terms in the sum:

.. math::


   2 \sin(\frac{1}{2} d) C = \\
   \bigg ( \sin(a + \frac{1}{2}d) - \sin(a - \frac{1}{2}d) \bigg ) + \\
   \bigg ( \sin(a + \frac{3}{2}d) - \sin(a + \frac{1}{2}d) \bigg ) + \\
   ... \\
   \bigg ( \sin(a + (N - \frac{3}{2}) d) - \sin(a + (N - \frac{5}{2} d) \bigg ) +
   \\
   \bigg ( \sin(a + (N - \frac{1}{2}) d) - \sin(a + (N - \frac{3}{2} d) \bigg )

The series telescopes, because the second term at each iteration cancels
the first term at the previous iteration. We are left only with the
first term in the last iteration and the second term from the first:

.. math::


   2 \sin(\frac{1}{2} d) C =
   \sin(a + (N - \frac{1}{2}) d) - \sin(a - \frac{1}{2} d)

Now we go the opposite direction with
:math:`\sin(\alpha + \beta) - \sin(\alpha - \beta) = 2\cos \alpha \sin \beta`.

Let :math:`\alpha + \beta = a + (N - \frac{1}{2}) d` and
:math:`\alpha - \beta = a - \frac{1}{2} d`. Solving for :math:`\alpha`
and :math:`\beta` we get:

.. math::


   2 \sin(\frac{1}{2} d) C =
   2 \cos( a + (N - 1) \frac{1}{2} d ) \sin( N \frac{1}{2} d )

We solve for :math:`C` to finish the proof. :math:`\blacksquare`

Sine sum
~~~~~~~~

This is almost identical, but applying:

.. math::


   \cos(\alpha + \beta) = \cos \alpha \cos \beta - \sin \alpha \sin \beta \\
   \cos(\alpha - \beta) = \cos \alpha \cos \beta + \sin \alpha \sin \beta \\
   \implies \\
   \cos(\alpha + \beta) - \cos(\alpha - \beta) = -2 \sin \alpha \sin \beta

Let:

.. math::


   S \triangleq \sum_{n=0}^{N-1}\sin(a + nd)

Only if :math:`\sin(\frac{1}{2}d) \ne 0`, we can safely multiply both
sides by :math:`-2 \sin(\frac{1}{2}d)` and continue with the same steps
as for the cosine:

.. math::


   -2 \sin(\frac{1}{2} d) S =
   \sum_{n=0}^{N-1}-2 \sin( a + nd ) \sin( \frac{1}{2} d ) \\
   = \sum_{n=0}^{N-1} \bigg ( \cos ( a + ( n + \frac{1}{2}) d )
   - \cos ( a + (n - \frac{1}{2}) d ) \bigg ) \\
   = \cos ( a + (N - \frac{1}{2}) d ) - \cos ( a - \frac{1}{2} d ) \\
   = -2 \sin (a + (N - 1)\frac{1}{2} d ) \sin ( N \frac{1}{2} d )

Then solve for :math:`S`.

:math:`\blacksquare`

