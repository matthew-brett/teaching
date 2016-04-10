########################
Vectors and dot products
########################

A vector is an ordered sequence of values:

.. math::

    \vec{v} = [ v_1, v_2, \cdots v_n ] \\

A vector can be *scaled* by a scalar $c$:

.. math::

    c \vec{v} \triangleq [ c v_1, c v_2, \cdots c v_n ]

Say we have two vectors containing $n$ values:

.. math::

    \vec{v} = [ v_1, v_2, \cdots v_n ] \\
    \vec{w} = [ w_1, w_2, \cdots w_n ]

Vector *addition* gives a new vector with $n$ values:

.. math::

    \vec{v} + \vec{w} \triangleq [ v_1 + w_1, v_2 + w_2, \cdots v_n + w_n ]

Vector addition is commutative because $v_i + w_i = w_i + v_i$:

.. math::

    \vec{v} + \vec{w} = \vec{w} + \vec{v}

The vector *dot product* is:

.. math::

    \vec{v} \cdot \vec{w} \triangleq \Sigma_{i=1}^n v_i w_i

We write the *length* of a vector $\vec{v}$ as $\VL{v}$:

.. math::

    \VL{v} \triangleq \sqrt{ \Sigma v_i^2 }

This is a generalization of Pythonagoras' theorem to $n$ dimensions.  For
example, the length of a two dimensional vector $[ x, y ]$ is the length of the
hypotenuse of the right-angle triangle formed by the points $(x, 0), (0, y),
(x, y)$.  This length is $\sqrt{x^2 + y^2}$.  For a point in three dimensions
${x, y, z}$, consider the right-angle triangle formed by $(x, y, 0), (0, 0,
z), (x, y, z)$.  The hypotenuse is length $\sqrt{\L{ [ x, y ] }^2 + z^2} =
\sqrt{ x^2 + y^2 + z^2 }$.

From the definition of vector length and the dot product, the square root of
the dot product of the vector with itself gives the vector length:

.. math::

    \VL{v} = \sqrt{ \vec{v} \cdot \vec{v} }

**************************
Properties of dot products
**************************

We will use the results from :doc:`some_sums`.

Commutative
===========

.. math::

    \vec{v} \cdot \vec{w} = \vec{w} \cdot \vec{v}

because $v_i w_i = w_i v_i$.

Distributive over vector addition
=================================

.. math::

    \vec{v} \cdot (\vec{w} + \vec{x}) = \vec{v} \cdot \vec{w} + \vec{v} \cdot
    \vec{x}

because:

.. math::

    \vec{v} \cdot (\vec{w} + \vec{x}) = \\
    \Sigma{ v_i ( w_i + x_i) } = \\
    \Sigma{ (v_i + w_i) } + \Sigma{ (v_i + x_i) } = \\
    \vec{v} \cdot \vec{w} + \vec{v} \cdot \vec{x}

Scalar multiplication
=====================

Say we have two scalars, $c$ and $d$:

.. math::

    (c \vec{v}) \cdot (d \vec{w}) = c d ( \vec{v} \cdot \vec{w} )

because:

.. math::

    (c \vec{v}) \cdot (d \vec{w}) = \\
    \Sigma{ c v_i d w_i } = \\
    c d \Sigma{ v_i w_i }


From the properties of distribution over addition and scalar multiplication:

.. math::

    \vec{v} \cdot (c \vec{w} + \vec{x}) = c (\vec{v} \cdot \vec{w}) + (\vec{v}
    \cdot \vec{x})

See: `properties of dot products <https://en.wikipedia.org/wiki/Dot_product#Properties>`_.

***********
Unit vector
***********

A unit vector is any vector with length 1.

To make a corresponding unit vector from any vector $\vec{v}$, divide by
$\VL{v}$:

.. math::

    \vec{u} = \frac{1}{ \VL{v} } \vec{v}

Let $g \triangleq \frac{1}{\VL{v}}$.  Then:

.. math::

    \L{ g \vec{v} }^2 = \\
    ( g \vec{v} ) \cdot ( g \vec{v} ) = \\
    g^2  \VL{v}^2 = 1


********************************************************
If two vectors are perpendicular, their dot product is 0
********************************************************

I based this proof on that in Gilbert Strang's "Introduction to Linear
Algebra" 4th edition, page 14.

Consider the triangle formed by the two vectors $\vec{v}$ and $\vec{w}$.  The
lengths of the sides of the triangle are $\VL{v}, \VL{w}, \L{\vec{v} -
\vec{w}}$.  When $\vec{v}$ and $\vec{w}$ are perpendicular, this is a
right-angled triangle with hypotenuse length $\L{\vec{v} - \vec{w}}$.  In this
situation, by Pythagoras:

.. math::

    \VL{v}^2 + \VL{w}^2 = \L{\vec{v} - \vec{w}}^2

Write the left hand side as:

.. math::

    \VL{v}^2 + \VL{w}^2 =
    v_1^2 + v_2^2 + \cdots v_n^2 +
    w_1^2 + w_2^2 + \cdots w_n^2

Write the right hand side as:

.. math::

    \L{\vec{v} - \vec{w}}^2 =
    (v_1^2 - 2v_1 w_1 + w_1^2) +
    (v_2^2 - 2v_2 w_2 + w_2^2) +
    \cdots
    (v_n^2 - 2v_n w_1 + w_n^2)

The $v_i^2$ and $w_i^2$ terms on left and right cancel, so:

.. math::

    \VL{v}^2 + \VL{w}^2 = \L{\vec{v} - \vec{w}}^2 \implies \\
    0 = 2(v_1 w_1 + v_2 w_2 + \cdots v_n w_n) \implies \\
    0 = \vec{v} \cdot \vec{w}

By the `converse of Pythagoras' theorem
<https://en.wikipedia.org/wiki/Pythagorean_theorem#Converse>`_, if $\VL{v}^2 +
\VL{w}^2 \ne \L{\vec{v} - \vec{w}}^2$ then vectors $\vec{v}$ and $\vec{w}$ do
not form a right angle and are not perpendicular.

.. include:: links_names.inc
