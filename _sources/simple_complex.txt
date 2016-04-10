############################
Refresher on complex numbers
############################

:math:`i` is the *imaginary unit*:

.. math::

   i \triangleq \sqrt{-1}

Therefore:

.. math::

   i^2 = -1

Engineers often write the imaginary unit as :math:`j`, and that is the
convention that Python uses:

.. nbplot::

    >>> 1j * 1j
    (-1+0j)

An imaginary number is a real number multiplied by the imaginary unit.  For
example :math:`3i` is an imaginary number.

A *complex* number is a number that has a *real* part and an *imaginary*
part. The real part is a real number. The imaginary part is an imaginary
number.

For example, consider the complex number :math:`a = (4 + 3i)`. :math:`a`
has two parts. The first is a real number, often written as
:math:`\R{a}`, and the second is an imaginary number :math:`\I{a}`:

.. math::

   \R{a} = 4 \\
   \I{a} = 3i

Now consider the complex number :math:`c = (p + qi)`.  :math:`\R{c} = p, \I{c}
= qi`.

To multiply a complex number :math:`c` by a real number :math:`r`, we
multiply both real and imaginary parts by the real number:

.. math::

   r (p + qi) = (rp + rqi)

.. nbplot::

    >>> a = (4 + 3j)
    >>> 2 * a
    (8+6j)

Multiplying a complex number by an imaginary number follows the same
logic, but remembering that :math:`i^2 = -1`. Let us say :math:`s i` is
our imaginary number:

.. math::


   si (p + qi) = (s i p + s i q i) \\
   = (s p i + s q i^2) \\
   = (s p i - s q) \\
   = (-s q + s p i)

.. nbplot::

    >>> 3j * a
    (-9+12j)
