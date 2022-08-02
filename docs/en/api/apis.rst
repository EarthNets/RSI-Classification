.. role:: hidden
    :class: hidden-section

rsicls.apis
===================================

These are some high-level APIs for classification tasks.

.. contents:: rsicls.apis
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: rsicls.apis

Train
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   init_random_seed
   set_random_seed
   train_model

Test
------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   single_gpu_test
   multi_gpu_test

Inference
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   init_model
   inference_model
   show_result_pyplot
