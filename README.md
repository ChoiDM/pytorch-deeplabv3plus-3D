# pytorch deeplabv3plus 3D
3D version of [Deeplabv3+](https://arxiv.org/abs/1802.02611)


## How to use
'''
from network.deeplabv3_3d import DeepLabV3_3D

num_classes = 10 # Number of classes. (= number of output channel)
input_channels = 3 # Number of input channel
resnet = 'resnet18_os16' # Base resnet architecture
last_activation = 'softmax' # 'softmax', 'sigmoid' or None

model = DeepLabV3_3D(num_classes = num_classes, input_channels = input_channels, resnet = resnet, last_activation = last_activation)
'''
