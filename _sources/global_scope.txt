##########################################
Global and local scope of Python variables
##########################################

See: `variables and scope`_ in the online `Python textbook`_.

.. nbplot::

    >>> # - compatibility with Python 2
    >>> from __future__ import print_function  # print('me') instead of print 'me'
    >>> from __future__ import division  # 1/2 == 0.5, not 0

The *scope* of a variable refers to the places that you can see or access a
variable.

If you define a variable at the top level of your script or module or
notebook, this is a global variable:

.. nbplot::

    >>> my_var = 3

The variable is *global* because any Python function or class defined in
this module or notebook, is able to access this variable. For example:

.. nbplot::

    >>> def my_first_func():
    ...     # my_func can 'see' the global variable
    ...     print('I see "my_var" = ', my_var, ' from "my_first_func"')

.. nbplot::

    >>> my_first_func()
    I see "my_var" =  3  from "my_first_func"

Variables defined inside a function or class, are not global. Only the
function or class can see the variable:

.. nbplot::

    >>> def my_second_func():
    ...     a = 10
    ...     print('I see "a" = ', a, 'from "my_second_func"')

.. nbplot::

    >>> my_second_func()
    I see "a" =  10 from "my_second_func"

But here, down in the top (global) level of the notebook, we can't see that
variable:

.. nbplot::
    :raises: NameError

    >>> a
    Traceback (most recent call last):
      ...
    NameError: name 'a' is not defined

The variable ``a`` is therefore said to be *local* to the function. Put
another way, the variable ``a`` has local scope. Conversely the variable
``my_var`` has global scope.

The full rules on variable scope cover many more cases than these simple
examples.

See this `notebook on variable scope`_ by `Sebastian Raschka`_ for a nice
walkthrough.

.. include:: links_names.inc
