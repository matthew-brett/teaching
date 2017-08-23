# $\newcommand{L}[1]{\| #1 \|}\newcommand{VL}[1]{\L{ \vec{#1} }}\newcommand{R}[1]{\operatorname{Re}\,(#1)}\newcommand{I}[1]{\operatorname{Im}\, (#1)}$
#
# ## Functions are objects
#
# In Python, functions are objects, like any other object.

# - compatibility with Python 2
from __future__ import print_function  # print('me') instead of print 'me'
from __future__ import division  # 1/2 == 0.5, not 0

# If I make a string in Python:

name = 'Matthew'

# then I have a Python object of type `str`:

type(name)

# Let’s say I defined a function `add`:

def add(a, b):
    return a + b

# Now I have another Python object, of type `function`:

type(add)

# With my string, I can refer to the same string object, with a different
# variable name:

prisoner = name
prisoner

# It’s the same for functions, because functions are objects too:

my_add = add
type(my_add)

# Functions are objects you can “call” by appending parentheses enclosing
# arguments you want to pass:

add(1, 2)

my_add(1, 2)

# As for any other object in Python, you can pass function objects to other
# functions:

def run_a_func(func, arg1, arg2):
    result = func(arg1, arg2)
    print('Result was', result)

run_a_func(add, 1, 2)

run_a_func(my_add, 1, 2)

def sub(a, b):
    return a - b

run_a_func(sub, 1, 2)
