Transform theory
^^^^^^^^^^^^^^^^
As light traverse the optical system, the relative position of the photons with respect to the optical axis change not only due to changes in the focal length, but also due to aberrations, reflections, etc. For the sake of simplicity, we can assume geometrical optics and understand the effects of the optical elements as $T: \mathbb R^2\to R^2$ functions between pairs of coordinates.

Transforms are bi-directional, with the forward direction being that that transforms object plane coordinates into image plane coordinates, and the backward direction that that transforms image plane coordinates into object plane coordinates. This implies that transforms must be *invertible*. Although theoretically :math:`T_b=T_f^{-1}` and therefore :math:`T_b\circ T_f=\mathbb I`, this is rarely achievable in the real world. In practical terms, it is required that the composition of the same transform in both backward and forward directions equals the identity transform up to certain (hopefully small) numerical error:

.. math::

   T_f\circ T_b=\mathbb I+e_{fb}

