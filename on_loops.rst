######################################
"for" and "while", "break" and "else:"
######################################

In :doc:`brisk_python`, we saw the use of ``break`` in ``for`` and ``while``
loops.

``for`` and ``while`` loops that use ``break``, can be followed by ``else:``
clauses.  The ``else:`` clause executes only when there was no ``break``
during the loop.

In the next fragment, we are doing an inefficient search for prime numbers
from 2 through 30.  In this basic ``for`` loop, we use the ``is_prime``
variable as a flag to indicate whether we have found the current number to be
prime:

.. nbplot::

    >>> primes = []
    >>> for x in range(2, 30):
    ...     # Assume x is prime until shown otherwise
    ...     is_prime = True
    ...     for p in primes:
    ...         # x exactly divisible by prime -> x not prime
    ...         if (x % p) == 0:
    ...             is_prime = False
    ...             break
    ...     if is_prime:
    ...         primes.append(x)
    ...
    >>> print("Primes in 2 through 30", primes)
    Primes in 2 through 30 [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

Using a flag variable like ``is_prime`` is a common pattern, so Python allows
us to do the same thing with an extra ``else:`` clause:

.. nbplot::

    >>> primes = []
    >>> for x in range(2, 30):
    ...     for p in primes:
    ...         # x exactly divisible by prime -> x not prime
    ...         if (x % p) == 0:
    ...             break
    ...     else:
    ...         # else: block executes if no 'break" in previous loop
    ...         primes.append(x)
    ...
    >>> print("Primes in 2 through 30", primes)
    Primes in 2 through 30 [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
