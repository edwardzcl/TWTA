# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:24:54 2018

@author: ZCL
"""

#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
- 1. This model has 1,068,298 paramters and Dorefa compression strategy(weight:1 bit, active: 1 bit),
after 500 epoches' training with GPU,accurcy of 41.1% was found.
- 2. For simplified CNN layers see "Convolutional layer (Simplified)"
in read the docs website.
- 3. Data augmentation without TFRecord see `tutorial_image_preprocess.py` !!
Links
-------
.. https://www.tensorflow.org/versions/r0.9/tutorials/deep_cnn/index.html
.. https://github.com/tensorflow/tensorflow/tree/r0.9/tensorflow/models/image/cifar10
Note
------
The optimizers between official code and this code are different.
Description
-----------
The images are processed as follows:
.. They are cropped to 24 x 24 pixels, centrally for evaluation or randomly for training.
.. They are approximately whitened to make the model insensitive to dynamic range.
For training, we additionally apply a series of random distortions to
artificially increase the data set size:
.. Randomly flip the image from left to right.
.. Randomly distort the image brightness.
.. Randomly distort the image contrast.
Speed Up
--------
Reading images from disk and distorting them can use a non-trivial amount
of processing time. To prevent these operations from slowing down training,
we run them inside 16 separate threads which continuously fill a TensorFlow queue.
"""

import os
import time
import tensorflow as tf
import tensorlayer as tl
import ast
import argparse

parser = argparse.ArgumentParser()
# resume from previous checkpoint after 100 epochs and decay learning rate by 10 times
parser.add_argument('--resume', type=ast.literal_eval, default=False)
# learning rate, # decay it by 10×  after 100 epochs, and set resume = True
parser.add_argument('--learning_rate', type=float, default=0.01)
# training or inference
parser.add_argument('--mode', type=str, default='training')
args = parser.parse_args()

print(args.resume, args.learning_rate, args.mode)

tf.reset_default_graph()

## Download data, and convert to TFRecord format, see ```tutorial_tfrecord.py```
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

print('X_train.shape', X_train.shape)  # (50000, 32, 32, 3)
print('y_train.shape', y_train.shape)  # (50000,)
print('X_test.shape', X_test.shape)  # (10000, 32, 32, 3)
print('y_test.shape', y_test.shape)  # (10000,)
print('X %s   y %s' % (X_test.dtype, y_test.dtype))


def data_to_tfrecord(images, labels, filename):
    """ Save data into TFRecord """
    if os.path.isfile(filename):
        print("%s exists" % filename)
        return
    print("Converting data into %s ..." % filename)
    # cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    for index, img in enumerate(images):
        img_raw = img.tobytes()
        ## Visualize a image
        # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        label = int(labels[index])
        # print(label)
        ## Convert the bytes back to image as follow:
        # image = Image.frombytes('RGB', (32, 32), img_raw)
        # image = np.fromstring(img_raw, np.float32)
        # image = image.reshape([32, 32, 3])
        # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                }
            )
        )
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
    if is_train ==True:
        # 1. Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [24, 24, 3])
        # 2. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)
        # 3. Randomly change brightness.
        img = tf.image.random_brightness(img, max_delta=63)
        # 4. Randomly change contrast.
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        # 5. Subtract off the mean and divide by the variance of the pixels.
        try:  # TF 0.12+
            img = tf.image.per_image_standardization(img)
        except Exception:  # earlier TF versions
            img = tf.image.per_image_whitening(img)

    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        try:  # TF 0.12+
            img = tf.image.per_image_standardization(img)
        except Exception:  # earlier TF versions
            img = tf.image.per_image_whitening(img)
    elif is_train == None:
        img = img

    label = tf.cast(features['label'], tf.int32)
    return img, label


## Save data into TFRecord files
data_to_tfrecord(images=X_train, labels=y_train, filename="train.cifar10")
data_to_tfrecord(images=X_test, labels=y_test, filename="test.cifar10")

batch_size = 200
model_file_name = "./model_cifar10_advanced.ckpt"
#resume = False  # load model, resume from previous checkpoint?

with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # prepare data in cpu
    x_train_, y_train_ = read_and_decode("train.cifar10", True)
    x_test_, y_test_ = read_and_decode("test.cifar10", False)
    # set the number of threads here
    x_train_batch, y_train_batch = tf.train.shuffle_batch(
        [x_train_, y_train_], batch_size=batch_size, capacity=2000, min_after_dequeue=1000, num_threads=32
    )
    # for testing, uses batch instead of shuffle_batch
    x_test_batch, y_test_batch = tf.train.batch(
        [x_test_, y_test_], batch_size=batch_size, capacity=50000, num_threads=32
    )

    def model(x_crop, y_, is_train, reuse):
        """ For more simplified CNN APIs, check tensorlayer.org """
        with tf.variable_scope("model", reuse=reuse):
            net = tl.layers.InputLayer(x_crop, name='input')
            net = tl.layers.Conv2d(net, 64, (3, 3), (1, 1), padding='SAME', b_init=None, name='conv00')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn00')
            net = tl.layers.TernaryLayer(net, name='ternary00')

            net = tl.layers.TernaryConv2d(net, n_filter=64, filter_size=(4, 4), strides=(1, 1), padding='SAME', b_init=None, name='conv0')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn0')
            net = tl.layers.TernaryLayer(net, name='ternary0')

            net = tl.layers.TernaryConv2d(net, n_filter=64, filter_size=(4, 4), strides=(1, 1), padding='SAME', b_init=None, name='conv1')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn1')
            net = tl.layers.TernaryLayer(net, name='ternary1')

            net = tl.layers.TernaryConv2d(net, n_filter=64, filter_size=(2, 2), strides=(2, 2), padding='SAME', b_init=None, name='conv2')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn2')
            net = tl.layers.TernaryLayer(net, name='ternary2')

            net = tl.layers.TernaryConv2d(net, n_filter=128, filter_size=(4, 4), strides=(1, 1), padding='SAME', b_init=None, name='conv3')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn3')
            net = tl.layers.TernaryLayer(net, name='ternary3')

            net = tl.layers.TernaryConv2d(net, n_filter=128, filter_size=(4, 4), strides=(1, 1), padding='SAME', b_init=None, name='conv4')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn4')
            net = tl.layers.TernaryLayer(net, name='ternary4')

            net = tl.layers.TernaryConv2d(net, n_filter=128, filter_size=(2, 2), strides=(2, 2), padding='SAME', b_init=None, name='conv5')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn5')
            net = tl.layers.TernaryLayer(net, name='ternary5')

            net = tl.layers.TernaryConv2d(net, n_filter=256, filter_size=(4, 4), strides=(1, 1), padding='SAME', b_init=None, name='conv6')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn6')
            net = tl.layers.TernaryLayer(net, name='ternary6')

            net = tl.layers.TernaryConv2d(net, n_filter=256, filter_size=(4, 4), strides=(1, 1), padding='SAME', b_init=None, name='conv7')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn7')
            net = tl.layers.TernaryLayer(net, name='ternary7')

            net = tl.layers.TernaryConv2d(net, n_filter=256, filter_size=(4, 4), strides=(2, 2), padding='VALID', b_init=None, name='conv8')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn8')
            net = tl.layers.TernaryLayer(net, name='ternary8')

            net = tl.layers.TernaryConv2d(net, n_filter=512, filter_size=(2, 2), strides=(1, 1), padding='VALID', b_init=None, name='conv9')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn9')
            net = tl.layers.TernaryLayer(net, name='ternary9')

            net = tl.layers.TernaryConv2d(net, n_filter=512, filter_size=(1, 1), strides=(1, 1), padding='VALID', b_init=None, name='conv10')
            net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train,  name='bn10')
            net = tl.layers.TernaryLayer(net, name='ternary10')
            

            net = tl.layers.FlattenLayer(net, name='flatten')

            # the last layer is usually high-precison refer to the previous works.
            net = tl.layers.DenseLayer(net, 10, b_init=None, name='output')

            y = net.outputs

            ce = tl.cost.cross_entropy(y, y_, name='cost')
            # L2 for the MLP, without this, the accuracy will be reduced by 15%.
            L2 = 0
            for p in tl.layers.get_variables_with_name('first/', True, True):
                L2 += tf.contrib.layers.l2_regularizer(0.00004)(p)       
            for q in tl.layers.get_variables_with_name('second/W', True, True):     
                L2 += tf.contrib.layers.l2_regularizer(0.00001)(q)   
            cost = ce + L2

            # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
            correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return net, cost, acc

    ## You can also use placeholder to feed_dict in data after using
    ## val, l = sess.run([x_train_batch, y_train_batch]) to get the data
    # x_crop = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
    # y_ = tf.placeholder(tf.int32, shape=[batch_size,])
    # cost, acc, network = model(x_crop, y_, None)

    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        network, cost, acc, = model(x_train_batch, y_train_batch, True, False)
        _, cost_test, acc_test = model(x_test_batch, y_test_batch, False, True)



    ## train
    n_epoch = 500
    #learning_rate = 0.01
    print_freq = 1
    n_step_epoch = int(len(y_train) / batch_size)
    n_step = n_epoch * n_step_epoch

    with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
        train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)

    tl.layers.initialize_global_variables(sess)
    if args.resume:
        print("Load existing model " + "!" * 10)
        saver = tf.train.Saver()
        saver.restore(sess, model_file_name)

    network.print_params(False)
    network.print_layers()

    print('   learning_rate: %f' % args.learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    if args.mode == 'training':
       for epoch in range(n_epoch):
           start_time = time.time()
           train_loss, train_acc, n_batch = 0, 0, 0
           for s in range(n_step_epoch):
               ## You can also use placeholder to feed_dict in data after using
               # val, l = sess.run([x_train_batch, y_train_batch])
               # tl.visualize.images2d(val, second=3, saveable=False, name='batch', dtype=np.uint8, fig_idx=2020121)
               # err, ac, _ = sess.run([cost, acc, train_op], feed_dict={x_crop: val, y_: l})
               err, ac, _ = sess.run([cost, acc, train_op])
               step += 1
               train_loss += err
               train_acc += ac
               n_batch += 1

           if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
               print(
                   "Epoch %d : Step %d-%d of %d took %fs" %
                   (epoch, step, step + n_step_epoch, n_step, time.time() - start_time)
               )
               print("   train loss: %f" % (train_loss / n_batch))
               print("   train acc: %f" % (train_acc / n_batch))

               test_loss, test_acc, n_batch = 0, 0, 0
               for _ in range(int(len(y_test) / batch_size)):
                   err, ac = sess.run([cost_test, acc_test])
                   test_loss += err
                   test_acc += ac
                   n_batch += 1
               print("   test loss: %f" % (test_loss / n_batch))
               print("   test acc: %f" % (test_acc / n_batch))

           if (epoch + 1) % (print_freq * 50) == 0:
               print("Save model " + "!" * 10)
               saver = tf.train.Saver()
               save_path = saver.save(sess, model_file_name)
               # you can also save model into npz
               tl.files.save_npz(network.all_params, name='model.npz', sess=sess)
               # and restore it as follow:
               # tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)

    else:
        test_loss, test_acc, n_batch = 0, 0, 0
        for _ in range(int(len(y_test) / batch_size)):
            err, ac = sess.run([cost_test, acc_test])
            test_loss += err
            test_acc += ac
            n_batch += 1
        print("   test loss: %f" % (test_loss / n_batch))
        print("   test acc: %f" % (test_acc / n_batch))

        coord.request_stop()
        coord.join(threads)
        sess.close()