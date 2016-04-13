########################################
Introducing principal component analysis
########################################

This page was largely inspired by these two excellent tutorials:

*  http://randomanalyses.blogspot.com/2012/01/principal-components-analysis.html
*  https://liorpachter.wordpress.com/2014/05/26/what-is-principal-component-analysis

Let's say I have some data in a 2D array :math:`\mathbf{X}`.

I have taken two different measures - or *variables* - and 50 samples.  So, I
have two variables for each of the 50 samples.

Each column is one sample (I have 50 columns). Each row is one variable (I
have two rows).

.. nbplot::

    >>> import numpy as np
    >>> import numpy.linalg as npl
    >>> # Make some random, but predictable data
    >>> np.random.seed(1966)
    >>> X = np.random.multivariate_normal([0, 0], [[3, 1.5], [1.5, 1]], size=50).T
    >>> X.shape
    (2, 50)

To make things simpler, I will subtract the mean across samples from each
variable:

.. nbplot::

    >>> # Subtract mean across samples (mean of each variable)
    >>> x_mean = X.mean(axis=1)
    >>> X[0] = X[0] - x_mean[0]
    >>> X[1] = X[1] - x_mean[1]

The values for the two variables (rows) in :math:`\mathbf{X}` are somewhat
correlated:

.. nbplot::

    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(X[0], X[1])
    <...>
    >>> plt.axis('equal')
    (...)

We want to explain the variation in these data.

The variation we want to explain is given by the sum of squares of the data
values.

.. nbplot::

    >>> squares = X ** 2
    >>> print(np.sum(squares))
    155.669289858

The sums of squares of the data can be thought of as the squared lengths of
the 50 2D vectors in the columns of :math:`\mathbf{X}`.

We can think of each sample as being a point on a 2D coordinate system, where
the first variable is the position on the x axis, and the second is the
position on the y axis. In fact, this is how we just plotted the values in the
scatter plot. We can also think of each column as a 2D *vector*. Call
:math:`\vec{v_j}` the vector contained in column :math:`j` of matrix
:math:`\mathbf{X}`, where :math:`j \in 1..50`.

The sum of squares across the variables, is also the squared distance of the
point (column) from the origin (0, 0). That is the same as saying that the sum
of squares is the squared *length* of :math:`\vec{v_j}`.  This can be written
as :math:`\|\vec{v_j}\|^2`

Take the first column / point / vector as an example (:math:`\vec{v_1}`):

.. nbplot::

    >>> v1 = X[:, 0]
    >>> v1
    array([ 3.378322,  2.068158])

.. nbplot::
    :include-source: false

    # Show first vector as sum of x and y axis vectors
    x, y = v1
    # Make subplots for vectors and text
    fig, (vec_ax, txt_ax) = plt.subplots(1, 2)
    font_sz = 18
    # Plot 0, 0
    vec_ax.plot(0, 0, 'ro')
    # Show vectors as arrows
    vec_ax.arrow(0, 0, x, 0, color='r', length_includes_head=True, width=0.01)
    vec_ax.arrow(0, 0, x, y, color='k', length_includes_head=True, width=0.01)
    vec_ax.arrow(x, 0, 0, y, color='b', length_includes_head=True, width=0.01)
    # Label origin
    vec_ax.annotate('$(0, 0)$', (-0.6, -0.7), fontsize=font_sz)
    # Label vectors
    vec_ax.annotate(r'$\vec{{v_1}} = ({x:.2f}, {y:.2f})$'.format(x=x, y=y),
                    (x / 2 - 2.2, y + 0.1), fontsize=font_sz)
    vec_ax.annotate(r'$\vec{{x}} = ({x:.2f}, 0)$'.format(x=x),
                    (x / 2 - 0.2, -0.7), fontsize=font_sz)
    vec_ax.annotate(r'$\vec{{y}} = (0, {y:.2f})$'.format(y=y),
                    (x + 0.2, y / 2 - 0.1), fontsize=font_sz)
    # Make sure axes are correct lengths
    vec_ax.axis((-1, 6, -1, 3))
    vec_ax.set_aspect('equal', adjustable='box')
    vec_ax.set_title(r'x- and y- axis components of $\vec{v_1}$')
    vec_ax.axis('off')
    # Text about lengths
    txt_ax.axis('off')
    txt_ax.annotate(r'$\|\vec{v_1}\|^2 = \|\vec{x}\|^2 + \|\vec{y}\|^2$ =' +
                    '\n' +
                    '${x:.2f}^2 + {y:.2f}^2$'.format(x=x, y=y),
                    (0.1, 0.45), fontsize=font_sz)

So, the sums of squares we are trying to explain can be expressed as the sum
of the squared distance of each point from the origin, where the points
(vectors) are the columns of :math:`\mathbf{X}`:

.. nbplot::

    >>> # Plot points and lines connecting points to origin
    >>> plt.scatter(X[0], X[1])
    <...>
    >>> for point in X.T:  # iterate over columns
    ...     plt.plot(0, 0)
    ...     plt.plot([0, point[0]], [0, point[1]], 'r:')
    [...]
    >>> plt.axis('equal')
    (...)

Put another way, we are trying to explain the squares of the lengths of the
dotted red lines on the plot.

At the moment, we have not explained anything, so our current unexplained sum
of squares is:

.. nbplot::

    >>> print(np.sum(X ** 2))
    155.669289858

For the following you will need to know how to use vector dot products to
project one vector on another. There is good background in `this Khan academy
video on projection
<https://www.khanacademy.org/math/linear-algebra/matrix_transformations/lin_trans_examples/v/introduction-to-projections>`__
if you need to revise that - and more background from the same series of
videos if you need to freshen up on `vector length
<https://www.khanacademy.org/math/linear-algebra/vectors_and_spaces/dot_cross_products/v/vector-dot-product-and-vector-length>`__,
`mathematical properties of dot products
<https://www.khanacademy.org/math/linear-algebra/vectors_and_spaces/dot_cross_products/v/vector-dot-product-and-vector-length>`__,
`unit vectors
<https://www.khanacademy.org/math/linear-algebra/matrix_transformations/lin_trans_examples/v/unit-vectors>`__
and `angles between vectors
<https://www.khanacademy.org/math/linear-algebra/vectors_and_spaces/dot_cross_products/v/defining-the-angle-between-vectors>`__.

Let us now say that we want to try and find a line that will explain the
maximum sum of squares in the data.

We define our line with a unit vector :math:`\hat{u}`. All points on the line
can be expressed with :math:`c\hat{u}` where :math:`c` is a scalar.

Our best fitting line :math:`c\hat{u}` is the line that comes closest to the
points, in the sense of minimizing the squared distance between the line and
points.

Put a little more formally, for each point :math:`\vec{v_j}` we will find the
distance :math:`d_j` between :math:`\vec{v_j}` and the line. We want the line
with the smallest :math:`\sum_j{d_j^2}`.

What do we mean by the *distance* in this case? The distance :math:`d_i` is
the distance between the point :math:`\vec{v_i}` and the projection of that
point onto the line :math:`c\hat{u}`. The projection of :math:`\vec{v_i}` onto
the line defined by :math:`\hat{u}` is, as we remember, given by
:math:`c\hat{u}` where :math:`c = \vec{v_i}\cdot\hat{u}`.

Looking at the scatterplot, we might consider trying a unit vector at 45
degrees angle to the x axis:

.. nbplot::

    >>> u_guessed = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    >>> u_guessed
    array([ 0.707107,  0.707107])

This is a unit vector:

.. nbplot::

    >>> np.sum(u_guessed ** 2)
    1.0

.. nbplot::

    >>> plt.scatter(X[0], X[1])
    <...>
    >>> plt.arrow(0, 0, u_guessed[0], u_guessed[1], width=0.01, color='r')
    <...>
    >>> plt.axis('equal')
    (...)
    >>> plt.title('Guessed unit vector')
    <...>

Let's project all the points onto that line:

.. nbplot::

    >>> u_guessed_row = u_guessed.reshape(1, 2)  # A row vector
    >>> c_values = u_guessed_row.dot(X)  # c values for scaling u
    >>> projected = u_guessed_row.T.dot(c_values)
    >>> # scale u by values to get projection
    >>> plt.scatter(X[0], X[1], label='actual')
    <...>
    >>> plt.scatter(projected[0], projected[1], color='r', label='projected')
    <...>
    >>> for i in range(X.shape[1]):
    ...     # Plot line between projected and actual point
    ...     proj_pt = projected[:, i]
    ...     actual_pt = X[:, i]
    ...     plt.plot([proj_pt[0], actual_pt[0]], [proj_pt[1], actual_pt[1]], 'k')
    [...]
    >>> plt.axis('equal')
    (...)
    >>> plt.legend(loc='upper left')
    <...>
    >>> plt.title("Actual and projected points for guessed $\hat{u}$")
    <...>

The projected points (in red), are the positions of the points that can be
explained by projection onto the guessed line defined by :math:`\hat{u}`. The
red projected points also have their own sum of squares:

.. nbplot::

    >>> print(np.sum(projected ** 2))
    133.381320743

Because we are projecting onto a unit vector, :math:`\|c\hat{u}\|^2 = c\hat{u}
\cdot c\hat{u} = c^2(\hat{u} \cdot \hat{u}) = c^2`.  Therefore the
``c_values`` are also the lengths of the projected vectors, so the sum of
squares of the ``c_values`` also gives us the sum of squares of the projected
points:

.. nbplot::

    >>> print(np.sum(c_values ** 2))
    133.381320743

As we will see later, this is the sum of squares from the original points that
have been explained by projection onto :math:`\hat{u}`.

Once I have the projected points, I can calculate the remaining distance of
the actual points from the projected points:

.. nbplot::

    >>> remaining = X - projected
    >>> distances = np.sqrt(np.sum(remaining ** 2, axis=0))
    >>> distances
    array([ 0.926426,  0.714267,  0.293125,  0.415278,  0.062126,  0.793188,
            0.684554,  1.686549,  0.340629,  0.006746,  0.301138,  0.405397,
            0.995828,  0.171356,  1.094742,  0.780583,  0.183566,  0.974734,
            0.732008,  0.495833,  0.96324 ,  1.362817,  0.262868,  0.092597,
            0.477803,  0.041519,  0.84133 ,  0.33801 ,  0.019824,  0.853356,
            0.069814,  0.244263,  0.347968,  0.470062,  0.705145,  1.173709,
            0.838709,  1.006069,  0.731594,  0.74943 ,  0.343281,  0.55684 ,
            0.287912,  0.479475,  0.977735,  0.064308,  0.127375,  0.157425,
            0.01017 ,  0.519997])

I can also express the overall (squared) remaining distance as the sum
of squares:

.. nbplot::

    >>> print(np.sum(remaining ** 2))
    22.2879691152

I'm going to try a whole lot of different values for :math:`\hat{u}`, so
I will make a function to calculate the result of projecting the data
onto a line defined by a unit vector :math:`\hat{u}`:

.. nbplot::

    >>> def line_projection(u, X):
    ...     """ Return columns of X projected onto line defined by u
    ...     """
    ...     u = u.reshape(1, 2)  # A row vector
    ...     c_values = u.dot(X)  # c values for scaling u
    ...     projected = u.T.dot(c_values)
    ...     return projected

Next a small function to return the vectors remaining after removing the
projections:

.. nbplot::

    >>> def line_remaining(u, X):
    ...     """ Return vectors remaining after removing cols of X projected onto u
    ...     """
    ...     projected = line_projection(u, X)
    ...     remaining = X - projected
    ...     return remaining

Using these little functions, I get the same answer as before:

.. nbplot::

    >>> print(np.sum(line_remaining(u_guessed, X) ** 2))
    22.2879691152

Now I will make lots of :math:`\hat{u}` vectors spanning half the circle:

.. nbplot::

    >>> angles = np.linspace(0, np.pi, 10000)
    >>> x = np.cos(angles)
    >>> y = np.sin(angles)
    >>> u_vectors = np.vstack((x, y))
    >>> u_vectors.shape
    (2, 10000)

.. nbplot::

    >>> plt.plot(u_vectors[0], u_vectors[1], '+')
    [...]
    >>> plt.axis('equal')
    (...)
    >>> plt.tight_layout()

I then get the remaining sum of squares after projecting onto each of these
unit vectors:

.. nbplot::

    >>> remaining_ss = []
    >>> for u in u_vectors.T: # iterate over columns
    ...     remaining = line_remaining(u, X)
    ...     remaining_ss.append(np.sum(remaining ** 2))
    >>> plt.plot(angles, remaining_ss)
    [...]
    >>> plt.xlabel('Angle of unit vector')
    <...>
    >>> plt.ylabel('Remaining sum of squares')
    <...>

It looks like the minimum value is for a unit vector at around angle 0.5
radians:

.. nbplot::

    >>> min_i = np.argmin(remaining_ss)
    >>> angle_best = angles[min_i]
    >>> print(angle_best)
    0.498620616186

.. nbplot::

    >>> u_best = u_vectors[:, min_i]
    >>> u_best
    array([ 0.878243,  0.478215])

.. nbplot::

    >>> plt.scatter(X[0], X[1])
    <...>
    >>> plt.arrow(0, 0, u_best[0], u_best[1], width=0.01, color='r')
    <...>
    >>> plt.axis('equal')
    (...)
    >>> plt.title('Best unit vector')
    <...>

Do the projections for this best line look better than before?

.. nbplot::

    >>> projected = line_projection(u_best, X)
    >>> plt.scatter(X[0], X[1], label='actual')
    <...>
    >>> plt.scatter(projected[0], projected[1], color='r', label='projected')
    <...>
    >>> for i in range(X.shape[1]):
    ...     # Plot line between projected and actual point
    ...     proj_pt = projected[:, i]
    ...     actual_pt = X[:, i]
    ...     plt.plot([proj_pt[0], actual_pt[0]], [proj_pt[1], actual_pt[1]], 'k')
    [...]
    >>> plt.axis('equal')
    (...)
    >>> plt.legend(loc='upper left')
    <...>
    >>> plt.title("Actual and projected points for $\hat{u_{best}}$")
    <...>

Now we have found a reasonable choice for our first best fitting line, we have
a set of remaining vectors that we have not explained. These are the vectors
between the projected and actual points.

.. nbplot::

    >>> remaining = X - projected
    >>> plt.scatter(remaining[0], remaining[1], label='remaining')
    <...>
    >>> plt.arrow(0, 0, u_best[0], u_best[1], width=0.01, color='r')
    <...>
    >>> plt.annotate('$\hat{u_{best}}$', u_best, xytext=(20, 20), textcoords='offset points', fontsize=20)
    <...>
    >>> plt.legend(loc='upper left')
    <...>
    >>> plt.axis('equal')
    (...)

Now it is obvious the next line we need to best explain the remaining sum of
squares. We want another unit vector orthogonal to the first.  This is because
we have already explained everything that can be explained along the direction
of :math:`\hat{u_{best}}`, and we only have two dimensions, so there is only
one remaining direction along which the variation can occur.

.. nbplot::

    >>> u_best_orth = np.array([np.cos(angle_best + np.pi / 2), np.sin(angle_best + np.pi / 2)])
    >>> plt.scatter(remaining[0], remaining[1], label='remaining')
    <...>
    >>> plt.arrow(0, 0, u_best[0], u_best[1], width=0.01, color='r')
    <...>
    >>> plt.arrow(0, 0, u_best_orth[0], u_best_orth[1], width=0.01, color='g')
    <...>
    >>> plt.annotate('$\hat{u_{best}}$', u_best, xytext=(20, 20), textcoords='offset points', fontsize=20)
    <...>
    >>> plt.annotate('$\hat{u_{orth}}$', u_best_orth, xytext=(20, 20), textcoords='offset points', fontsize=20)
    <...>
    >>> plt.axis('equal')
    (...)

Now the projections onto :math:`\hat{u_{orth}}` are the same as the
remaining points, because the remaining points already lie along the
line defined by :math:`\hat{u_{orth}}`.

.. nbplot::

    >>> projected_onto_orth = line_projection(u_best_orth, remaining)
    >>> np.allclose(projected_onto_orth, remaining)
    True

If we have really found the line :math:`\hat{u_{best}}` that removes the most
sum of squares from the remaining points, then this is the *first principal
component* of :math:`\mathbf{X}`. :math:`\hat{u_{orth}}` will be the second
principal component of :math:`\mathbf{X}`.

Now for a trick. Remember that the two principal components are orthogonal to
one another. That means, that if I project the data onto the second principal
component :math:`\hat{u_{orth}}`, I will (by the definition of orthogonal)
pick up no component of the columns of :math:`\mathbf{X}` that is colinear
(predictable via projection) with :math:`\hat{u_{best}}`.

This means that I can go straight to the projection onto the second component,
from the original array :math:`\mathbf{X}`.

.. nbplot::

    >>> # project onto second component direct from data
    >>> projected_onto_orth_again = line_projection(u_best_orth, X)
    >>> # Gives same answer as projecting remainder from first component
    >>> np.allclose(projected_onto_orth_again - projected_onto_orth, 0)
    True

For the same reason, I can calculate the projection coefficients :math:`c` for
both components at the same time, by doing matrix multiplication:

.. nbplot::

    >>> # Components as rows in a 2 by 2 array
    >>> components = np.vstack((u_best, u_best_orth))
    >>> components
    array([[ 0.878243,  0.478215],
           [-0.478215,  0.878243]])

.. nbplot::

    >>> # Calculating projection coefficients with array dot
    >>> c_values = components.dot(X)
    >>> # Result of projecting on first component, via array dot
    >>> u = u_best.reshape(1, 2)  # first component as row vector
    >>> c = c_values[0].reshape(1, 50)  # c for first component as row vector
    >>> projected_1 = u.T.dot(c)
    >>> # The same as doing the original calculation
    >>> np.allclose(projected_1, line_projection(u_best, X))
    True

.. nbplot::

    >>> # Result of projecting on second component, via array dot
    >>> u = u_best_orth.reshape(1, 2)  # second component as row vector
    >>> c = c_values[1].reshape(1, 50)  # c for second component as row vector
    >>> projected_2 = u.T.dot(c)
    >>> # The same as doing the original calculation
    >>> np.allclose(projected_2, line_projection(u_best_orth, X))
    True

**************************************************************
The principal component lines are new axes to express the data
**************************************************************

My original points were expressed in the orthogonal, standard x and y axes. My
principal components give new orthogonal axes. When I project, I have just
re-expressed my original points on these new orthogonal axes. Let's call the
projections of :math:`\vec{v_1}` onto the first and second components:
:math:`proj_1\vec{v_1}`, :math:`proj_2\vec{v_1}`.

For example, here is my original first point :math:`\vec{v_1}` expressed using
the projections onto the principal component axes:

.. nbplot::
    :include-source: false

    # Show v1 as sum of projections onto components 1 and 2
    x, y = v1
    # Projections onto first and second component
    p1_x, p1_y = projected_1[:, 0]
    p2_x, p2_y = projected_2[:, 0]
    # Make subplots for vectors and text
    fig, (vec_ax, txt_ax) = plt.subplots(1, 2)
    # Show 0, 0
    vec_ax.plot(0, 0, 'ro')
    # Show vectors with arrows
    vec_ax.arrow(0, 0, p1_x, p1_y, color='r', length_includes_head=True, width=0.01)
    vec_ax.arrow(0, 0, x, y, color='k', length_includes_head=True, width=0.01)
    vec_ax.arrow(p1_x, p1_y, p2_x, p2_y, color='b', length_includes_head=True, width=0.01)
    # Label origin
    vec_ax.annotate('$(0, 0)$', (-0.6, -0.7), fontsize=font_sz)
    # Label vectors
    vec_ax.annotate(r'$\vec{{v_1}} = ({x:.2f}, {y:.2f})$'.format(x=x, y=y),
                    (x / 2 - 2.2, y + 0.3), fontsize=font_sz)
    vec_ax.annotate(('$proj_1\\vec{{v_1}} = $\n'
                     '$({x:.2f}, {y:.2f})$').format(x=p1_x, y=p1_y),
                    (p1_x / 2 - 0.2, p1_y / 2 - 1.8), fontsize=font_sz)
    vec_ax.annotate(('$proj_2\\vec{{v_1}} =$\n'
                     '$({x:.2f}, {y:.2f})$').format(x=p2_x, y=p2_y),
                    (x + 0.3, y - 1.2), fontsize=font_sz)
    # Make sure axes are right lengths
    vec_ax.axis((-1, 6.5, -1, 3))
    vec_ax.set_aspect('equal', adjustable='box')
    vec_ax.set_title(r'first and and second principal components of $\vec{v_1}$')
    vec_ax.axis('off')
    # Text about length
    txt_ax.axis('off')
    txt_ax.annotate(
        r'$\|\vec{v_1}\|^2 = \|proj_1\vec{v_1}\|^2 + \|proj_2\vec{v_1}\|^2$ =' +
        '\n' +
        '${p1_x:.2f}^2 + {p1_y:.2f}^2 + {p2_x:.2f}^2 + {p2_y:.2f}^2$'.format(
        p1_x=p1_x, p1_y=p1_y, p2_x=p2_x, p2_y=p2_y),
        (0, 0.5), fontsize=font_sz)

We have re-expressed :math:`\vec{v_1}` by two new orthogonal vectors
:math:`proj_1\vec{v_1}` plus :math:`proj_2\vec{v_1}`. In symbols:
:math:`\vec{v_1} = proj_1\vec{v_1} + proj_2\vec{v_1}`.

The sum of component 1 projections and the component 2 projections add up to
the original vectors (points).

Sure enough, if I sum up the data projected onto the first component and the
data projected onto the second, I get back the original data:

.. nbplot::

    >>> np.allclose(projected_1 + projected_2, X)
    True

Doing the sum above is the same operation as matrix multiplication of the
transpose of the components with the projection coefficients (seeing that this
is so involves writing out a few cells of the matrix multiplication in symbols
and staring at it for a while):

.. nbplot::

    >>> data_again = components.T.dot(c_values)
    >>> np.allclose(data_again, X)
    True

********************************************
The components partition the sums of squares
********************************************

Notice also that I have partititioned the sums of squares of the data into a
part that can be explained by the first component, and a part that can be
explained by the second:

.. nbplot::

    >>> # Total sum of squares
    >>> print(np.sum(X ** 2))
    155.669289858

.. nbplot::

    >>> # The data projected onto the first component
    >>> proj_onto_first = line_projection(u_best, X)
    >>> # The data projected onto the second component
    >>> proj_onto_second = line_projection(u_best_orth, X)
    >>> # Sum of squares in the projection onto the first
    >>> ss_in_first = np.sum(proj_onto_first ** 2)
    >>> # Sum of squares in the projection onto the second
    >>> ss_in_second = np.sum(proj_onto_second ** 2)
    >>> # They add up to the total sum of squares
    >>> print((ss_in_first, ss_in_second, ss_in_first + ss_in_second))
    (143.97317154347922, 11.696118314873956, 155.66928985835318)

Why is this?

Consider the first vector in :math:`\mathbf{X}` : :math:`\vec{v_1}`. We have
re-expressed the length of :math:`\vec{v_1}` with the squared length of
:math:`proj_1\vec{v_1}` plus the squared length of :math:`proj_2\vec{v_1}`.
The length of :math:`\vec{v_1}` is unchanged, but we now have two new
orthogonal vectors making up the sides of the right angled triangle of which
:math:`\vec{v_1}` is the hypotenuse. The total sum of squares in the data is
given by:

.. math::

   \sum_j x^2 + \sum_j y^2 = \\
   \sum_j \left( x^2 + y^2 \right) = \\
   \sum_j \|\vec{v_1}\|^2 = \\
   \sum_j \left( \|proj_1\vec{v_1}\|^2 + \|proj_2\vec{v_1}\|^2 \right) = \\
   \sum_j \|proj_1\vec{v_1}\|^2 + \sum_j \|proj_2\vec{v_1}\|^2 \\

where :math:`j` indexes samples - :math:`j \in 1..50` in our case.

The first line shows the partition of the sum of squares into standard x and y
coordinates, and the last line shows the partition into the first and second
principal components.

*****************************************
Finding the principal components with SVD
*****************************************

You now know what a principal component analysis is.

It turns out there is a much quicker way to find the components than the slow
and dumb search that I did above.

For reasons that we don't have space to go into, we can get the components
using `Singular Value Decomposition
<https://en.wikipedia.org/wiki/Singular_value_decomposition>`__ (SVD) of
:math:`\mathbf{X}`.

See http://arxiv.org/abs/1404.1100 for more detail. :math:`\newcommand{\X}{\mathbf{X}}\newcommand{\U}{\mathbf{U}}\newcommand{\S}{\mathbf{\Sigma}}\newcommand{\V}{\mathbf{V}}`

The SVD on an array containing only real (not complex) values such as
:math:`\mathbf{X}` is defined as:

.. math::

   \X = \U \Sigma \V^T

If $\X$ is shape $M$ by $N$ then $\U$ is an $M$ by $M$ `orthogonal
matrix <https://en.wikipedia.org/wiki/Orthogonal_matrix>`__, $\S$ is a
`diagonal matrix <https://en.wikipedia.org/wiki/Diagonal_matrix>`__ shape $M$
by $N$, and $\V^T$ is an $N$ by $N$ orthogonal matrix.

.. nbplot::

    >>> U, S, VT = npl.svd(X)

The components are in the columns of the returned matrix $\U$.

.. nbplot::

    >>> U
    array([[-0.878298, -0.478114],
           [-0.478114,  0.878298]])

Remember that a vector :math:`\vec{r}` defines the same line as the
vector :math:`-\vec{r}`, so we do not care about a flip in the sign of
the principal components:

.. nbplot::

    >>> u_best
    array([ 0.878243,  0.478215])

.. nbplot::

    >>> u_best_orth
    array([-0.478215,  0.878243])

The returned vector ``S`` gives the :math:`M` `singular
values <https://en.wikipedia.org/wiki/Singular_value>`__ that form the
main diagonal of the $M$ by $N$ diagonal matrix $\S$. The values in ``S`` give
the square root of the explained sum of squares for each component:

.. nbplot::

    >>> S ** 2
    array([ 143.973173,   11.696117])

The SVD is quick to compute for ``X``, but notice that the returned
array ``VT`` is $N$ by $N$, and isn't of much use to us for our PCA:

.. nbplot::

    >>> VT.shape
    (50, 50)

In fact we can get our $\U$ and $\S$ without calculating $\V^T$ by doing SVD
on the variance / covariance matrix of the variables. If $M$ is much smaller
than $N$ this saves a lot of time and memory.

Here's why that works:

.. math::

   \U \S \V^T = \X \\
   (\U \S \V^T)(\U \S \V^T)^T = \X \X^T

By the matrix transpose rule and associativity of matrix multiplication:

.. math::

   \U \S \V^T \V \S^T \U^T = \X \X^T

By the definition of the SVD, $\V^T$ is an orthogonal matrix, so $\V$ is
the inverse of $\V^T$ and $\V^T \V = I$. $\S$ is a diagonal
matrix so $\S \S^T = \S^2$, where $\S^2$ is a square diagonal matrix shape
$M$ by $M$ containing the squares of the singular values from $\S$:

.. math::

   \U \S^2 \U^T = \X \X^T

This last formula is the formula for the SVD of $\X \X^T$. So, we can get our
$\U$ and $\S$ from the SVD on $\X \X^T$.

.. nbplot::

    >>> # Finding principal components using SVD on X X^T
    >>> unscaled_cov = X.dot(X.T)
    >>> U_vcov, S_vcov, VT_vcov = npl.svd(unscaled_cov)
    >>> U_vcov
    array([[-0.878298, -0.478114],
           [-0.478114,  0.878298]])

We know from the derivation above that ``VT_vcov`` is just the transpose of
$\U$:

.. nbplot::

    >>> np.allclose(U, VT_vcov.T)
    True

The returned vector ``S_vcov`` from the SVD on $\X \X^T$ now contains the
explained sum of squares for each component:

.. nbplot::

    >>> S_vcov
    array([ 143.973173,   11.696117])

Sums of squares and variance from PCA
-------------------------------------

We have done the SVD on the *unscaled* variance / covariance matrix.
*Unscaled* means that the values in the matrix have not been divided by
:math:`N`, or :math:`N-1`, where :math:`N` is the number of samples.  This
matters little for our case, but sometimes it is useful to think in terms of
the variance explained by the components, rather than the sums of squares.

The standard *variance* of a vector :math:`\vec{x}` with :math:`N`
elements :math:`x_1, x_2, ... x_N` indexed by :math:`i` is given by
:math:`\frac{1}{N-1} \sum_i \left( x_i - \bar{x} \right)^2`.
:math:`\bar{x}` is the mean of :math:`\vec{x}`:
:math:`\bar{x} = \frac{1}{N} \sum_i x_i`. If :math:`\vec{q}` already has
zero mean, then the variance of :math:`\vec{q}` is also given by
:math:`\frac{1}{N-1} \vec{q} \cdot \vec{q}`.

The :math:`N-1` divisor for the variance comes from `Bessel's
correction <http://en.wikipedia.org/wiki/Bessel%27s_correction>`__ for
bias.

The covariance between two vectors :math:`\vec{x}, \vec{y}` is
:math:`\frac{1}{N-1} \sum_i \left( x_i - \bar{x} \right) \left( y_i - \bar{y} \right)`.
If vectors :math:`\vec{q}, \vec{p}` already both have zero mean, then
the covariance is given by :math:`\frac{1}{N-1} \vec{q} \cdot \vec{p}`.

Our unscaled variance covariance has removed the mean and done the dot
products above, but it has not applied the :math:`\frac{1}{N-1}`
scaling, to get the true variance / covariance.

For example, the standard numpy covariance function ``np.cov`` completes
the calculation of true covariance by dividing by :math:`N-1`.

.. nbplot::

    >>> # Calculate unscaled variance covariance again
    >>> unscaled_cov = X.dot(X.T)
    >>> # When divided by N-1, same as result of 'np.cov'
    >>> N = X.shape[1]
    >>> np.allclose(unscaled_cov / (N - 1), np.cov(X))
    True

We could have run our SVD on the true variance covariance matrix. The
result would give us exactly the same components. This might make sense
from the fact that the lengths of the components are always scaled to 1
(unit vectors):

.. nbplot::

    >>> scaled_U, scaled_S, scaled_VT = npl.svd(np.cov(X))
    >>> np.allclose(scaled_U, U), np.allclose(scaled_VT, VT_vcov)
    (True, True)

The difference is only in the *singular values* in the vector ``S``:

.. nbplot::

    >>> S_vcov
    array([ 143.973173,   11.696117])

.. nbplot::

    >>> scaled_S
    array([ 2.938228,  0.238696])

As you remember, the singular values from the unscaled covariance matrix were
the sum of squares explained by each component. The singular values from the
true covariance matrix are the *variances* explained by each component. The
variances are just the sum of squares divided by the correction in the
denominator, in our case, :math:`N-1`:

.. nbplot::

    >>> S_vcov / (N - 1)
    array([ 2.938228,  0.238696])

So far we have described the PCA as breaking up the sum of squares into parts
explained by the components. If we do the SVD on the true covariance matrix,
then we can describe the PCA as breaking up the *variance* of the data (across
samples) into parts explained by the components. The only difference between
these two is the scaling of the ``S`` vector.
