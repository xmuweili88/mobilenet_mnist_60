import mobilenet_v1
import numpy as np
import os
import tensorflow.contrib.slim as slim
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


# 1. Loads data set.
mnist = input_data.read_data_sets('MNIST_data')

# 2. Defines network.
# 2.1 Input feature dim = Nx784, N is sample number
x = tf.placeholder(tf.float32, [None, 784], name='x-input')

# 2.2 Reshapes the feature to Nx28x28 images
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 2.3 Defines the network.
with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):
    y, _ = mobilenet_v1.mobilenet_v1(
        x_image,
        num_classes=10,
        is_training=True,
    )
# 2.4 Input ground truth labels in on-hot.
y_ = tf.placeholder(tf.int32, [None, 10], name='y-input')

# 2.5 Defines Loss function.
loss = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=y_)

# 2.6 Defines accuracy.
accuracy = tf.metrics.accuracy(labels=tf.argmax(y_, axis=1),
                               predictions=tf.argmax(y, axis=1))
global_step = tf.Variable(0, trainable=False)

# 2.7 Train operation is minimizing the loss.
train_operation = tf.train.AdamOptimizer(1e-3)\
    .minimize(loss, global_step=global_step)

# 3. Trains the network
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    # Trains 10000 steps.
    for i in range(10000):
        # Each step uses 100 training samples.
        xs, ys = mnist.train.next_batch(100)
        # Converts the labels into probability vectors.
        ys_h = get_one_hot(ys, 10)
        # Runs the training operation.
        _, loss_value, accuracy_value, step = \
            sess.run([train_operation, loss, accuracy, global_step], feed_dict={x: xs, y_: ys_h})

        print("After %d training step(s), "
              "on training batch, loss is %g. "
              "accuracy is = %g"
              % (step, loss_value, accuracy_value[0]))

        # Saves the model into the disk.
        if i % 1000 == 0:
            saver.save(sess,
                       os.path.join('./mobilenet_v1/', 'model.ckpt'),
                       global_step=global_step)






