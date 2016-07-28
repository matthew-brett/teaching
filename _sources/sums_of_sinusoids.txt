#################
Sums of sinusoids
#################

This page largely based on http://math.stackexchange.com/a/1239123 with
thanks.

Paraphrasing `Wolfwram mathworld
<http://mathworld.wolfram.com/Sinusoid.html>`_ - a *sinusoid* is a function of
some variable, say $t$, that is similar to the sine function but may be
shifted in phase, frequency, amplitude, or any combination of the three.

The `general formula for a sinusoid function
<https://en.wikipedia.org/wiki/Sine_wave>`_ is:

.. math::
    :label: sin_sinusoid

    f(t) = A \sin(2 \pi f t + \theta) = A \sin(\omega t + \theta)

where:

* $A$ is the *amplitude* |--| the maximum value of the function;
* $f$ is the *ordinary frequency* |--| the number of cycles per unit time;
* $\omega = 2 \pi f$ is the *angular frequency* |--| the number of radians
  per unit time;
* $\theta$ is the *phase offset* (in radians).

The standard sine function $f(t) = \sin(\omega t)$ is a special case of a sinusoid,
with $A = 1$, $\theta = 0$.

The standard cosine function $f(t) = \cos(\omega t)$ is a special case of a sinusoid,
with $A = 1$, $\theta = -\pi / 2$.

Remembering :doc:`angle_sum`, we can write any sinusoid as a sum of a sine and
a cosine:

.. math::
    :label: sinusoid_as_sum

    A \sin(\omega t + \theta) \iff \\
    A \sin(\omega t) \cos(\theta) + A \cos(\omega t) \sin(\theta) \iff \\
    A' \sin(\omega t) + A'' \cos(\omega t)

Where $A' = A \cos(\theta)$ and $A'' = A \sin(\theta)$.  Equation
:eq:`sinusoid_as_sum` also shows that any sum of a sine and cosine can be
written as a single sinusoid.  Thus, any sum of sinusoids, *of the same input*
$\omega t$, is also a sinusoid:

.. math::

    A \sin(\omega t + \theta) + B \sin(\omega t + \phi) =
    (A' + B') \sin(\omega t) - (A'' + B'') \cos(\omega t)

.. include:: links_names.inc
