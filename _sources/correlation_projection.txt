##########################
Correlation and projection
##########################

Here we phrase the `Pearson product-moment correlation coefficient`_ in terms
of vectors.

Say we have two vectors of $n$ values:

.. math::

    \vec{x} = [x_1, x_2, ... , x_n]

    \vec{y} = [y_1, y_2, ... , y_n]

Write the mean of the values in $\vec{x}$ as $\bar{x}$:

.. math::

    \bar{x} = \frac{1}{n} \sum_{i=1}^n x_i

Define two new vectors, $\vec{x^c}, \vec{y^c}$ that contain the values in
$\vec{x}, \vec{y}$ with their respective means subtracted:

.. math::

    \vec{x^c} = [x_1 - \bar{x}, x_2 - \bar{x}, ... , x_n - \bar{x}]

    \vec{y^c} = [y_1 - \bar{y}, y_2 - \bar{y}, ... , y_n - \bar{y}]

Define the sample *variance* of $\vec{x}$ as the mean of the
squared deviations from the mean:

.. math::

    v_x = \frac{1}{n} \sum (x_1 - \bar{x})^2

The sample standard deviation of $\vec{x}$:

.. math::

    s_x = \sqrt{v_x}

Now define the `standardized <https://en.wikipedia.org/wiki/Standard_score>`_
versions of $\vec{x}, \vec{y}$ as:

.. math::

    \vec{x^z} = \frac{1}{s_x} \vec{x^c}

    \vec{y^z} = \frac{1}{s_y} \vec{y^c}

The Pearson product-moment correlation coefficient of $\vec{x}, \vec{y}$ is
given by $1 / n$ times the dot product of $\vec{x^z}, \vec{y^z}$:

.. math::

    r_{xy} = \frac{1}{n} \vec{x^z} \cdot \vec{y^z}

The equivalent expression in terms of sums rather than vectors is:

.. math::

    r_{xy} =\frac{1}{n} \sum ^n _{i=1} \left( \frac{x_i - \bar{x}}{s_x}
    \right) \left( \frac{y_i - \bar{y}}{s_y} \right)

Rewrite in terms of $\vec{x^c}, \vec{y^c}$ (see :doc:`on_vectors`):

.. math::

    r_{xy} = \frac{1}{n} \frac{\vec{x^c} \cdot \vec{y^c}}
            {s_x s_y}

    s_x = \sqrt{ \frac{1}{n} \vec{x^c} \cdot \vec{x^c} }
        = \sqrt{\frac{1}{n}} \; \VL{x^c}

    s_y = \sqrt{\frac{1}{n}} \; \VL{y^c}

    r_{xy} = \frac{\vec{x^c} \cdot \vec{y^c}} {\VL{x^c} \VL{y^c}}

The Pearson product-moment correlation coefficient is the dot product between
the vectors $\vec{x^c}, \vec{y^c}$ after normalizing the vectors to unit length.

$r_{xy}$ is therefore :doc:`the cosine of the angle <vector_angles>` between
$\vec{x^c}$ and $\vec{y^c}$.

.. include:: links_names.inc
