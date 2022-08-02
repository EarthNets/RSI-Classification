.. role:: hidden
    :class: hidden-section

rsicls.models.utils
===================================

This package includes some helper functions and common components used in various networks.

.. contents:: rsicls.models.utils
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: rsicls.models.utils

Common Components
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   InvertedResidual
   SELayer
   ShiftWindowMSA
   MultiheadAttention
   ConditionalPositionEncoding

Helper Functions
------------------

channel_shuffle
^^^^^^^^^^^^^^^
.. autofunction:: channel_shuffle

make_divisible
^^^^^^^^^^^^^^
.. autofunction:: make_divisible

to_ntuple
^^^^^^^^^^^^^^
.. autofunction:: to_ntuple
.. autofunction:: to_2tuple
.. autofunction:: to_3tuple
.. autofunction:: to_4tuple

is_tracing
^^^^^^^^^^^^^^
.. autofunction:: is_tracing
