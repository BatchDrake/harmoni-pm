Integration and energy transport
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We refer by *integration* to the calculation of the number of photons that arrive to each pixel in a given period of time, assuming an intensity pattern at the other end of the optical system. This calculation is implemented the :code:`ImageSampler` component and is based on radiative transfer. Let :math:`dE_t` the energy differential corresponding to the amount of radiative energy coming from a region in the sky of radiative intensity :math:`I` and angular size :math:`d\Omega` entering an aperture of area :math:`A` in a time :math:`dt` and a small range of frequencues :math:`d\nu`. This quantity can be written as:

  .. math::

     dE_t = Id\Omega A\text{cos }\theta d\nu dt

Since the field is relatively small, :math:`\text{cos }\theta\approx1` for all the rays that fall in the detector. On the other hand, the amount of energy :math:`dE_d` arriving to a point in the detector at the instrument's focal plane will be a fraction of the total energy that entered the telescope due to opacities, obstructions and aperture stops. This dependence can be written in terms of a dimensionless efficiency factor :math:`\beta`:

   .. math::

      dE_d = \beta dE_t

*TODO: consider* :math:`e^{-\tau}`!

Let :math:`F` the flux corresponding to the region :math:`d\Omega` that is focused in a small area :math:`dS` in the focal plane. The energy differential in the detector will be:

   .. math::

      dE_d = FdSd\nu dt

which allows us to deduce the following expression for the flux in the focal plane:

   .. math::

      F = \beta IA \frac{d\Omega}{dS} = \beta\frac{IA}{f^2}

where :math:`f` is the total focal length, if we assume aberration-free optics. If the power spectral density of the light is known, we can calculate the photon flux by:

   .. math::

      Q = \int F \frac{d\nu}{h\nu}
      
which, multiplied by the pixel area :math:`A_p` and assuming that the CCD Nyquist-samples the image plane, provides the number of photons per unit time striking a pixel. The true number of photons that arrived to a specific pixel in an integration time :math:`\Delta t` can be modelled as a Poisson-distributed random variable with :math:`\lambda=QA_p\Delta t`.


Focal distance and aberrations
------------------------------
In the previous point, we stated that :math:`\frac{d\Omega}{dS}=\frac{1}{f^2}` *if we assume aberration-free optics*. This will rarely be the case in the real world, and therefore a proper understanding of the relationship between aberrations and focal distance is necessary.

Let :math:`T_b(\vec{x})=[\phi(\vec{x}), \psi(\vec{x})]` be the compound backward transform of all the optical elements in the optical path, where :math:`\vec{x}=x\hat{e}_x+y\hat{e}_y` is the position vector of a point over the focal plane and :math:`\phi(\vec{x})` and :math:`\psi(\vec{x})` are two angular coordinates of the corresponding point in the sky with respect to the telescope pointing and centered in the equator. In this system :math:`\phi` is the longitude and :math:`\psi` its latitude. The orthonormal basis :math:`\hat{e}_x,\hat{e}_y` can be assumed to be conveniently aligned to the rows and columns of pixels of the detector. 

The solid angle differential :math:`d\Omega=\text{cos }\psi d\phi d\psi` is related to the focal plane area differential :math:`dS=dxdy`:

.. math ::
   \text{cos }\psi d\phi d\psi = \text{det }J_b dxdy

Where :math:`J_b` is the Jacobian matrix of the backward transform:

.. math ::
   J_b=\begin{pmatrix} \frac{\partial\phi}{\partial x} & \frac{\partial\phi}{\partial y} \\
   \frac{\partial\psi}{\partial x} & \frac{\partial\psi}{\partial y}
   \end{pmatrix}

In the paraxial approximation, :math:`\text{cos }\psi\approx1`. :math:`J_b` allows a polar decomposition of the form :math:`AR`, where :math:`A` is a symmetric matrix and :math:`R` a rotation matrix that can be connected to field rotation and the specific pick-off arm configuration. In the absence of rotations :math:`J_b=A` and in the absence of aberrations we can choose :math:`\phi` so that it only depends on :math:`x` and :math:`\psi` only depends on :math:`y`, which  implies that :math:`\frac{\partial\phi}{\partial y} = \frac{\partial\psi}{\partial x}=0` and :math:`A` will be simplified down to a diagonal matrix:

.. math ::
   \frac{d\Omega}{dS} = \text{det }AR = \text{det A} = \frac{\partial \phi}{\partial x}\frac{\partial\psi}{\partial y} = \frac{1}{f^2}

which is the ideal case assumed earlier. As radial aberrations are introduced, :math:`A` will lose diagonality and the determinant will change. This relationship suggests a way to estimate the focal distances from the coordinate transform and measure the degree of aberration in the optics:

.. math ::
   E(\vec{x}) = ||1 - f^2 \text{det }\tilde{J}_b(\vec{x})||

with :math:`E` a dimensionless quality figure that measures how much the aberrations are affecting the image in a given point :math:`\vec{x}` in the focal plane, :math:`f` the theoretical focal distance and :math:`\tilde{J}_b(\vec{x})` a numerical calculation of the Jacobian of the transform in the surroundings of :math:`\vec{x}`.

Oversampling
---------------
Coordinate transforms must be calculated at least once for every pixel in the detector. If the detector Nyquist-samples the image plane, it should be enough to measure the flux at a fixed relative offset in every pixel. However, sub-Nyquist structure may exist in the image plane and oversampling would be needed to prevent aliasing / Moiré. 

For a given oversampling value :math:`M`, ``ImageSampler`` will measure the intensity of the image plane in :math:`M^2` points inside the pixel surface and calculate the average intensity of all of them. The offset of the sampling point :math:`0 \leq i, j < M` from the bottom left corner of a pixel of width :math:`\Delta x` and height :math:`\Delta y` will be:

.. math ::
   \vec{p}(i, j) = \left(i+\frac12\right)\frac{\Delta x}{M}\hat{e}_x+\left(j+\frac12\right)\frac{\Delta y}M\hat{e}_y

Implementation
~~~~~~~~~~~~~~
From the implementation perspective, the generation of these subpixel coordinates is vectorized and performed in several steps:

1. **Generation of the matrix of row indices.** A square matrix :math:`r_{ij}=i` is created as the tensor product of a ``numpy.linspace(0, M - 1, M)`` by a column vector of ones of the same size.
2. **Generation of the x offsets.** The vector :math:`x_k=[f(r_{ij})_k + \frac12]\delta x` is created, with :math:`f` the ``flatten`` operator (which returns a vector of :math:`M^2` components such that :math:`f(r_{ij})_k=r_{k\text{ mod }M,\lfloor k/M\rfloor}`) and :math:`\delta x = \Delta x/M`.
3. **Generation of the y offsets.** The vector :math:`y_k=[f(r^T_{ij})_k + \frac12]\delta y` is created, with :math:`\delta y = \Delta y/M`.
4. **Generation of the coordinate offset vector.** The matrix :math:`\mathtt{xy}_{k2}` is created, such that :math:`\mathtt{xy}_{k0}=x_k` and :math:`\mathtt{xy}_{k1}=y_k`.

The resulting :math:`\mathtt{xy}` vector is then added to the coordinate of the bottom-left corner of each pixel to derive the full oversampling points.

Slicing and evaluation
---------------------------
The integration of the light arriving at :math:`h\times w` pixels with an oversampling of :math:`M` requires the calculation of :math:`h\times w\times M^2` coordinate transforms. In addition to the computatational cost of this operation, the memory allocation required by the vectorization of these operations may be too big to fit in the computer's memory. These two problems are addressed by slicing and parallelization.

Slicing refers to the process of selecting subsets of pixels in the detector from which integration operations are vectorized in ``numpy`` arrays. The maximum size of a slice is that of a contiguous square of side ``HARMONI_IMAGE_SAMPLER_SLICE_SIZE`` pixels, currently 128. The actual size of the slice may be smaller near the right and upper edges of the detector. The total number of coordinate transforms to perform can be calculated by multiplying the number of pixels in a slice :math:`K` by the square of the oversampling :math:`M^2`.

Details on coordinate generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The generation of coordinates is vectorized as well, and is performed in two stages. In a first stage (implemented in ``ImageSampler._integrate_slice``), matrices of row indices :math:`r_{ij}` and column indices :math:`c_{ij}` are created using the same stratregy as described in the oversampling implementation, and flattened in order to obtain a :math:`\mathtt{ij}_{k2}` matrix of :math:`K` rows in which each row represents the row and column indices of the pixels to integrate.

In a second stage (implemented in ``ImageSampler._integrate_pixels``, called by ``ImageSampler._integrate_slice``), the full coordinate list including subsampling is generated from an initial pixel list :math:`\mathtt{ij}_{k2}`. This stage is divided again in the following steps:

1. **Tiling of the oversampling offset matrix.** A matrix :math:`\mathtt{p\_{xy}}` consisting of concatenating :math:`\mathtt{xy}` :math:`K` times is generated, producing an oversampling offset matrix of :math:`KM^2` rows.
2. **Repetition of the pixel index matrix.** Each individual row of the index matrix :math:`\mathtt{ij}_{k2}` is repeated contiguosly :math:`M^2` times, growing the matrix up to :math:`KM^2` rows.
3. **Generation of the sampling point matrix.** The sampling coordinates are generated by adding the following matrix to :math:`\mathtt{p\_{xy}}`:

.. math ::

   \Delta\mathtt{p\_{xy}}_{k2}= \mathtt{ij}_{k2}\begin{pmatrix}\Delta x & 0 \\ 0 & \Delta y\end{pmatrix}+\vec{x}_0

where :math:`\vec{x}_0` is a position vector that encodes the physical displacement of the detector.

Evaluation
~~~~~~~~~~
The evaluation of the intensity field involves a data reduction due to oversampling, as the resulting intensity vector must match the original pixel index matrix passed to ``ImageSampler._integrate_pixels``. The intensity vector returned is evaluated by averaging the intensities belonging to the same pixel:

.. math ::

   \bar{I}_k = \frac{1}{M^2}\sum_{i=0}^{M-1} I[T_b(\mathtt{p\_xy}_{kM+i,0}, \mathtt{p\_xy}_{kM+i,1})]

where :math:`T_b` is the backward transform, :math:`\bar{I}_k` the reduced intensity vector and :math:`I` the intensity function of the image plane.

Parallelization
~~~~~~~~~~~~~~~
TODO: include details on thread parallelization

Time integration
----------------
TODO: include details on Poisson sampling, etc.


