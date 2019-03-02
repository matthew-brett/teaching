############################
Inverse of a diagonal matrix
############################

Let's say we have a shape (2, 2) diagonal matrix:

.. math::

    \newcommand{A}{\boldsymbol A}
    \newcommand{AI}{\boldsymbol A^{-1}}
    \newcommand{L}{\boldsymbol L}
    \newcommand{I}{\boldsymbol I}
    \A =
    \begin{bmatrix}
    p & 0 \\
    0 & q
    \end{bmatrix}

We want to find $\AI$, the left inverse of $\A$, such that:

.. math::

    \I = \AI \A

where $\I$ is a shape (2, 2) identity matrix:

.. math::

    \I =
    \begin{bmatrix}
    1 & 0  \\
    0 & 1
    \end{bmatrix}

As always when trying to find the inverse, we are solving a system of
simultaneous equations.   In the case of a diagonal matrix, the equations are
easier to solve because of the zeros off the diagonal.

Write $\AI$ as:

.. math::

    \AI =
    \begin{bmatrix}
    a & b  \\
    c & d
    \end{bmatrix}

From the definition of matrix multiplication, we now have:

.. math::

    \I = \AI \A =
    \begin{bmatrix}
    a & b  \\
    c & d
    \end{bmatrix}
    \begin{bmatrix}
    p & 0 \\
    0 & q
    \end{bmatrix}
    =
    \begin{bmatrix}
    a p + b 0 & a 0 + b q \\
    c p + d 0 & c 0 + d q
    \end{bmatrix}

Comparing $\I$ with this result, we have:

.. math::

    1 = a p + b 0 \\
    0 = a 0 + b q \\
    0 = c p + d 0 \\
    1 = c 0 + d q

From the central two equations, $b = c = 0$.  Substituting, we have:

.. math::

    1 = a p \implies a = \frac{1}{p} \\
    1 = d q \implies d = \frac{1}{q} \\

.. math::

    \AI =
    \begin{bmatrix}
    \frac{1}{p} & 0  \\
    0 & \frac{1}{q}
    \end{bmatrix}

This result generalizes for any shape ($n, n$) diagonal matrix.  To see this,
consider a shape (3, 3) or larger diagonal matrix, its inverse and the
identity.  Each zero in the identity matrix requires a zero in the
corresponding position in the inverse, otherwise the corresponding row /
column dot product will pick up a non-zero value from the diagonal.

.. math::

    \boldsymbol D =
    \begin{bmatrix}
    d_1 & 0 & 0 & ... & 0 \\
    0 & d_2 & 0 & ... & 0 \\
    0 & 0 & d_3 & ... & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & ... & d_n
    \end{bmatrix}

.. math::

    \boldsymbol D^{-1} =
    \begin{bmatrix}
    \frac{1}{d_1} & 0 & 0 & ... & 0 \\
    0 & \frac{1}{d_2} & 0 & ... & 0 \\
    0 & 0 & \frac{1}{d_3} & ... & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & ... & \frac{1}{d_n}
    \end{bmatrix}
