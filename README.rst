======================
 Pytorch Deeplabv3+ 3D
======================
This is PyTorch implementation of 3D `Deeplabv3+ <https://arxiv.org/abs/1802.02611>`_

Reference 1. https://github.com/jfzhang95/pytorch-deeplab-xception

Reference 2. https://github.com/fregu856/deeplabv3

---------------
How to use
---------------
.. code-block:: python


   from network.deeplabv3_3d import DeepLabV3_3D
   
   num_classes = 10 # Number of classes. (= number of output channel)
   input_channels = 3 # Number of input channel
   resnet = 'resnet18_os16' # Base resnet architecture ('resnet18_os16', 'resnet34_os16', 'resnet50_os16', 'resnet101_os16', 'resnet152_os16', 'resnet18_os8', 'resnet34_os18')
   last_activation = 'softmax' # 'softmax', 'sigmoid' or None
   
   model = DeepLabV3_3D(num_classes = num_classes, input_channels = input_channels, resnet = resnet, last_activation = last_activation)

