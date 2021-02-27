Common
^^^^^^^^^^^^^^^^^^^^
The common component represents the set of generic utility classes used across the project, like float array, exceptions and quantities.

.. class:: FloatArray(shape, buffer = None, offset = 0, strides = None, order = None)

   Convenience subclass of :code:`numpy.ndarray` with helper methods that ensures all arrays created from miscellaneous Python objects like tuples or lists are of the same :code:`ARRAY_TYPE` type. Currently, :code:`ARRAY_TYPE` is defined as :code:`float32`, giving a 24-bit significand (slightly more than 7 decimals of precission). This class also provides a unified way to define tensors in :code:`harmoni-pm`

   Constructor parameters are that of the underlying :code:`numpy.ndarray` constructor with :code:`dtype` set to :code:`ARRAY_TYPE`.

  .. staticmethod:: make(lst)

     Factory method that instantiates a :code:`FloatArray` from a given Python list or tuple specified by :code:`lst`. This :code:`lst` is passed directly to the underlying :code:`numpy.ndarray` constructor, and therefore the usual list/tuple - :code:`numpy.ndarray` conversion rules apply.
			     

  .. staticmethod:: compatible_with(arr)

     Returns `True` if `arr` is derived from `numpy.ndarray` and its `dtype` is set to `ARRAY_TYPE`. Otherwise, return false.

.. class:: Configuration

.. class:: InvalidPrototypeError

.. class:: InvalidTensorShapeError

.. class:: QuantityType
