#####################
Functions are objects
#####################

In Python, functions are objects, like any other object.

.. nbplot::

    >>> # - compatibility with Python 2
    >>> from __future__ import print_function  # print('me') instead of print 'me'
    >>> from __future__ import division  # 1/2 == 0.5, not 0

If I make a string in Python:

.. nbplot::

    >>> name = 'Matthew'

then I have a Python object of type ``str``:

.. nbplot::

    >>> type(name)
    <class 'str'>

Let's say I defined a function ``add``:

.. nbplot::

    >>> def add(a, b):
    ...     return a + b

Now I have another Python object, of type ``function``:

.. nbplot::

    >>> type(add)
    <class 'function'>

With my string, I can refer to the same string object, with a different
variable name:

.. nbplot::

    >>> prisoner = name
    >>> prisoner
    'Matthew'

It's the same for functions, because functions are objects too:

.. nbplot::

    >>> my_add = add
    >>> type(my_add)
    <class 'function'>

Functions are objects you can "call" by appending parentheses enclosing
arguments you want to pass:

.. nbplot::

    >>> add(1, 2)
    3

.. nbplot::

    >>> my_add(1, 2)
    3

As for any other object in Python, you can pass function objects to other
functions:

.. nbplot::

    >>> def run_a_func(func, arg1, arg2):
    ...     result = func(arg1, arg2)
    ...     print('Result was', result)

.. nbplot::

    >>> run_a_func(add, 1, 2)
    Result was 3

.. nbplot::

    >>> run_a_func(my_add, 1, 2)
    Result was 3

.. nbplot::

    >>> def sub(a, b):
    ...     return a - b

.. nbplot::

    >>> run_a_func(sub, 1, 2)
    Result was -1
