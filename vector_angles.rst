######################
Angles between vectors
######################

Consider two vectors $\vec{w}$ and $\vec{v}$.

.. image:: images/vector_projection.*
    :height: 500
    :width: 400
    :scale: 300

We already know from :doc:`vector_projection` that the vector projection of
$\vec{w}$ onto $\vec{v}$ is $c \vec{v}$ where:

.. math::

    c = \frac{\vec{w} \cdot \vec{v}}{\VL{v}^2}

We know from the definition of projection that $c \vec{v}$ and $\vec{w} - c
\vec{v}$ form a right angle.

If the angle between $\vec{v}$ and $\vec{w}$ is $\alpha$ radians, then:

.. math::

    \VL{w} cos(\alpha) = \L{ c \vec{v} }
    = c \VL{v}
    = \frac{\vec{w} \cdot \vec{v}}{\VL{v}}

and:

.. math::

    \VL{v} \VL{w} cos(\alpha) = \vec{w} \cdot \vec{v}

********
Also see
********

* :doc:`on_vectors`;
* :doc:`vector_angles`.
* An alternative proof using the `Law of Cosines
  <https://en.wikipedia.org/wiki/Law_of_cosines>`_ in this `Khan academy video
  on angles between vectors
  <https://www.khanacademy.org/math/linear-algebra/vectors_and_spaces/dot_cross_products/v/defining-the-angle-between-vectors>`_.

.. include:: links_names.inc
