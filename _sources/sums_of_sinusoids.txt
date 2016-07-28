#################
Sums of sinusoids
#################

See: http://math.stackexchange.com/a/1239123

Paraphrasing `Wolfwram mathworld
<http://mathworld.wolfram.com/Sinusoid.html>`_ - a *sinusoid* is a function of
some variable, say $t$, that is similar to the sine function but may be
shifted in phase, frequency, amplitude, or any combination of the three.

The `general formula for a sinusoid function
<https://en.wikipedia.org/wiki/Sine_wave>`_ is:

.. math::
    :label: sin_sinusoid

    f(t) = A \sin(2 \pi f t + \psi) = A \sin(\omega t + \psi)

where:

* $A$ is the *amplitude* |--| the maximum value of the function;
* $f$ is the *ordinary frequency* |--| the number of cycles per unit time;
* $\omega = 2 \pi f$ is the *angular frequency* |--| the number of radians
  per unit time;
* $\psi$ is the *phase offset* (in radians).

The standard sine function $f(t) = \sin(\omega t)$ is a special case of a sinusoid,
with $A = 1$, $\psi = 0$.

The standard cosine function $f(t) = \cos(\omega t)$ is a special case of a sinusoid,
with $A = 1$, $\psi = -\pi / 2$.

Because $sin(x) = \cos(x - \pi / 2)$, we can also write any sinusoid as:

.. math::
    :label: cos_sinusoid

    f(t) = A \cos(\omega t + \theta)

where $\theta = \psi - \pi / 2$ from equation :eq:`sin_sinusoid`.

Remembering :doc:`angle_sum`, we can write any sinusoid as a sum of a sine and
a cosine.  Choosing the cosine version of the sinusoid formula (equation
:eq:`cos_sinusoid`):

.. math::
    :label: sinusoid_as_sum

    A \cos(\omega t + \theta) \iff \\
    A \cos(\omega t) \cos(\theta) - A \sin(\omega t) \sin(\theta) \iff \\
    A' \cos(\omega t) - A'' \sin(\omega t)

Where $A' = A \cos(\theta)$ and $A'' = A \sin(\theta)$.  Equation
:eq:`sinusoid_as_sum` also shows that any sum of a sine and cosine can be
written as a single sinusoid.  Thus, any sum of sinusoids, *of the same input*
$\omega t$, is also a sinusoid:

.. math::

    A \cos(\omega t + \theta) + B \cos(\omega t + \phi) =
    (A' + B') \cos(\omega t) - (A'' + B'') \sin(\omega t)

.. include:: links_names.inc
