.. role:: hidden
    :class: hidden-section

rsicls.datasets
===================================

The ``datasets`` package contains several usual datasets for image classification tasks and some dataset wrappers.

.. currentmodule:: rsicls.datasets

Custom Dataset
--------------

.. autoclass:: CustomDataset

ImageNet
--------

.. autoclass:: ImageNet

.. autoclass:: ImageNet21k

CIFAR
-----

.. autoclass:: CIFAR10

.. autoclass:: CIFAR100

MNIST
-----

.. autoclass:: MNIST

.. autoclass:: FashionMNIST

VOC
---

.. autoclass:: VOC

Base classes
------------

.. autoclass:: BaseDataset

.. autoclass:: MultiLabelDataset

Dataset Wrappers
----------------

.. autoclass:: ConcatDataset

.. autoclass:: RepeatDataset

.. autoclass:: ClassBalancedDataset
