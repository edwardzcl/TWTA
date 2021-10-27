# -*- coding: utf-8 -*-
"""
Resnet-18 for ImageNet.

Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper 鈥淰ery Deep Convolutional Networks for
Large-Scale Image Recognition鈥? . The model achieves 92.7% top-5 test accuracy in ImageNet,
which is a dataset of over 14 million images belonging to 1000 classes.

Download Pre-trained Model
----------------------------
- Model weights in this example - vgg16_weights.npz : http://www.cs.toronto.edu/~frossard/post/vgg16/
- Caffe VGG 16 model : https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
- Tool to convert the Caffe models to TensorFlow's : https://github.com/ethereon/caffe-tensorflow

Note
------
- For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
- When feeding other images to the model be sure to properly resize or crop them
beforehand. Distorted images might end up being misclassified. One way of safely
feeding images of multiple sizes is by doing center cropping.
"""

import os
import numpy as np
import tensorflow as tf
from .. import _logging as logging
from ..layers import (Conv2d, DorefaConv2d, DorefaDenseLayer, DenseLayer, FlattenLayer, InputLayer, BatchNormLayer, ConcatLayer, ElementwiseLayer, SignLayer)
from ..files import maybe_download_and_extract, assign_params

__all__ = [
    'Resnet18_full_conv2',
]


def Resnet18_full_conv2(x_crop, y_, pretrained=False, end_with='fc1000', n_classes=1000, is_train=True, reuse=False, name=None):
    """Pre-trained MobileNetV1 model (static mode). Input shape [?, 224, 224, 3].
    To use pretrained model, input should be in BGR format and subtracted from ImageNet mean [103.939, 116.779, 123.68].

    Parameters
    ----------
    pretrained : boolean
        Whether to load pretrained weights. Default False.
    end_with : str
        The end point of the model [conv, depth1, depth2 ... depth13, globalmeanpool, out].
        Default ``out`` i.e. the whole model.
    n_classes : int
        Number of classes in final prediction.
    name : None or str
        Name for this model.

    Examples
    ---------
    Classify ImageNet classes, see `tutorial_models_resnet50.py`

    >>> # get the whole model with pretrained weights
    >>> resnet = tl.models.ResNet50(pretrained=True)
    >>> # use for inferencing
    >>> output = resnet(img1, is_train=False)
    >>> prob = tf.nn.softmax(output)[0].numpy()

    Extract the features before fc layer
    >>> resnet = tl.models.ResNet50(pretrained=True, end_with='5c')
    >>> output = resnet(img1, is_train=False)

    Returns
    -------
        ResNet50 model.

    """
    with tf.variable_scope("model", reuse=reuse):
         net = InputLayer(x_crop, name="input")
         net = Conv2d(net, 32, (3, 3), (1, 1), padding='SAME', b_init=None, name='conv00')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn00')

         net = Conv2d(net, 64, (3, 3), (1, 1), padding='SAME', b_init=None, name='conv0')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn0')
         #net = SignLayer(net)
         #脉冲化表达，同时还可以考虑这里是否需要高精度脉冲，或者减少层数，或者实现只量化权值的类型

         net1 = Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(2, 2), padding='SAME', b_init=None, name='stage1conv1')
         net1 = BatchNormLayer(net1, act=tf.nn.relu, is_train=is_train,  name='stage1bn1')
         #net1 = SignLayer(net1)
         net1 = Conv2d(net1, n_filter=64, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='stage1conv2')
         net1 = BatchNormLayer(net1, act=tf.identity, is_train=is_train,  name='stage1bn2')
         shortcut = Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(2, 2), padding='SAME', b_init=None, name='shortcut1conv1')
         shortcut = BatchNormLayer(shortcut, act=tf.identity, is_train=is_train,  name='stage1bn3')
         net1 = ElementwiseLayer([shortcut, net1], combine_fn=tf.add, act=None, name='elementwise1')
         net1.outputs = tf.nn.relu(net1.outputs)
         #也确实不太适合在这里加shortcut，一来妨碍BN_ReLu的融合，二来在这里加，已经在上面的BN层的ReLu之后，并且妨碍下一层的限幅
         #net1 = SignLayer(net1)

         shortcut = net1
         net2 = Conv2d(net1, n_filter=64, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='stage2conv1')
         net2 = BatchNormLayer(net2, act=tf.nn.relu, is_train=is_train,  name='stage2bn1')
         #net2 = SignLayer(net2)
         net2 = Conv2d(net2, n_filter=64, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='stage2conv2')
         #shortcut = DorefaConv2d(net, n_filter=64, filter_size=(3, 3), strides=(2, 2), padding='SAME', b_init=None, name='shortcut2conv1')
         net2 = BatchNormLayer(net2, act=tf.identity, is_train=is_train,  name='stage2bn2')
         net2 = ElementwiseLayer([shortcut, net2], combine_fn=tf.add, act=None, name='elementwise2')
         net2.outputs = tf.nn.relu(net2.outputs)
         #也确实不太适合在这里加shortcut，一来妨碍BN_ReLu的融合，二来在这里加，已经在上面的BN层的ReLu之后，并且妨碍下一层的限幅
         #net2 = SignLayer(net2)


         net3 = Conv2d(net2, n_filter=128, filter_size=(3, 3), strides=(2, 2), padding='SAME', b_init=None, name='stage3conv1')
         net3 = BatchNormLayer(net3, act=tf.nn.relu, is_train=is_train,  name='stage3bn1')
         #net3 = SignLayer(net3)
         net3 = Conv2d(net3, n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='stage3conv2')
         net3 = BatchNormLayer(net3, act=tf.identity, is_train=is_train,  name='stage3bn2')
         shortcut = Conv2d(net2, n_filter=128, filter_size=(3, 3), strides=(2, 2), padding='SAME', b_init=None, name='shortcut3conv1')
         shortcut = BatchNormLayer(shortcut, act=tf.identity, is_train=is_train,  name='stage3bn3')
         net3 = ElementwiseLayer([shortcut, net3], combine_fn=tf.add, act=None, name='elementwise3')
         net3.outputs = tf.nn.relu(net3.outputs)
         #也确实不太适合在这里加shortcut，一来妨碍BN_ReLu的融合，二来在这里加，已经在上面的BN层的ReLu之后，并且妨碍下一层的限幅
         #net3 = SignLayer(net3)

         shortcut = net3
         net4 = Conv2d(net3, n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='stage4conv1')
         net4 = BatchNormLayer(net4, act=tf.nn.relu, is_train=is_train,  name='stage4bn1')
         #net4 = SignLayer(net4)
         net4 = Conv2d(net4, n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='stage4conv2')
         #shortcut = DorefaConv2d(net2, n_filter=128, filter_size=(3, 3), strides=(2, 2), padding='SAME', b_init=None, name='shortcut4conv1')
         net4 = BatchNormLayer(net4, act=tf.identity, is_train=is_train,  name='stage4bn3')
         net4 = ElementwiseLayer([shortcut, net4], combine_fn=tf.add, act=None, name='elementwise4')
         net4.outputs = tf.nn.relu(net4.outputs)
         #也确实不太适合在这里加shortcut，一来妨碍BN_ReLu的融合，二来在这里加，已经在上面的BN层的ReLu之后，并且妨碍下一层的限幅
         #net4 = SignLayer(net4)

         net5 = Conv2d(net4, n_filter=256, filter_size=(3, 3), strides=(2, 2), padding='SAME', b_init=None, name='stage5conv1')
         net5 = BatchNormLayer(net5, act=tf.nn.relu, is_train=is_train,  name='stage5bn1')
         #net5 = SignLayer(net5)
         net5 = Conv2d(net5, n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='stage5conv2')
         net5 = BatchNormLayer(net5, act=tf.identity, is_train=is_train,  name='stage5bn2')
         shortcut = Conv2d(net4, n_filter=256, filter_size=(3, 3), strides=(2, 2), padding='SAME', b_init=None, name='shortcut5conv1')
         shortcut = BatchNormLayer(shortcut, act=tf.identity, is_train=is_train,  name='stage5bn3')
         net5 = ElementwiseLayer([shortcut, net5], combine_fn=tf.add, act=None, name='elementwise5')
         net5.outputs = tf.nn.relu(net5.outputs)
         #也确实不太适合在这里加shortcut，一来妨碍BN_ReLu的融合，二来在这里加，已经在上面的BN层的ReLu之后，并且妨碍下一层的限幅
         #net5 = SignLayer(net5)    

         shortcut = net5
         net6 = Conv2d(net5, n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='stage6conv1')
         net6 = BatchNormLayer(net6, act=tf.nn.relu, is_train=is_train,  name='stage6bn1')
         #net6 = SignLayer(net6)
         net6 = Conv2d(net6, n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='stage6conv2')
         #shortcut = DorefaConv2d(net4, n_filter=256, filter_size=(3, 3), strides=(2, 2), padding='SAME', b_init=None, name='shortcut6conv1')
         net6 = BatchNormLayer(net6, act=tf.identity, is_train=is_train,  name='stage6bn2')
         net6 = ElementwiseLayer([shortcut, net6], combine_fn=tf.add, act=None, name='elementwise6')
         net6.outputs = tf.nn.relu(net6.outputs)
         #也确实不太适合在这里加shortcut，一来妨碍BN_ReLu的融合，二来在这里加，已经在上面的BN层的ReLu之后，并且妨碍下一层的限幅
         #net6 = SignLayer(net6)     

         net7 = Conv2d(net6, n_filter=512, filter_size=(3, 3), strides=(1, 1), padding='VALID', b_init=None, name='conv7')
         net7 = BatchNormLayer(net7, act=tf.nn.relu, is_train=is_train,  name='bn7')
         #也确实不太适合在这里加shortcut，一来妨碍BN_ReLu的融合，二来在这里加，已经在上面的BN层的ReLu之后，并且妨碍下一层的限幅
         #似乎最后一层加了量化效果还更好
         #net7 = SignLayer(net7)
         #或许相比原始ResNet18还可以加层，或者直接改为全局/平均池化，联系到了脉冲      

    return net7




   
