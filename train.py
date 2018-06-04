import tensorflow as tf
from lenet import lenet5
from lenet import lenet5_parameter
#import lenet5
#import lenet5_parameter
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time

mnist_path = "mnist/"
#mnist_path = "oss://hualangdeeplearning.oss-cn-shanghai-internal.aliyuncs.com/cat_and_dog/mnist/"

mnist = input_data.read_data_sets(mnist_path, one_hot=True)
global_step = tf.Variable(0)
xs = tf.placeholder(tf.float32, lenet5_parameter.TRAIN_SIZE)
ys = tf.placeholder(tf.float32, lenet5_parameter.OUTPUT_SIZE)

prediction = lenet5.lenet5_model(xs)

# print(prediction)
soft_max_result = tf.nn.softmax(prediction)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(soft_max_result)))
softmax_result = tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction)
cross_entropy = tf.reduce_mean(softmax_result)
weight_loss = tf.add_n(tf.get_collection("loss"))
#loss = cross_entropy + weight_loss
loss = cross_entropy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys, 1), tf.argmax(soft_max_result, 1)), tf.float32))

decayed_learning_rate = tf.train.exponential_decay(1e-4, global_step = global_step, decay_steps=200, decay_rate=0.96)
train_step = tf.train.AdamOptimizer(decayed_learning_rate).minimize(loss)

init = tf.global_variables_initializer()
train_saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for i in list(range(30000)):
        x_batch, y_batch = mnist.train.next_batch(lenet5_parameter.BATCH_SIZE)
        x_batch = tf.reshape(x_batch, [lenet5_parameter.BATCH_SIZE, 28,28,1])
        x_batch = tf.pad(x_batch,[[0,0], [2,2], [2,2], [0,0]])
        x_batch = x_batch.eval()



        _,_, predict, cro_en, accum, soft = sess.run(
            [global_step, train_step, prediction, cross_entropy, accuracy, soft_max_result],
            feed_dict={xs: x_batch, ys: y_batch})
        if i % 50 == 0:

            print("time:", time.time())
            print("step: %d" % i)
            #print("prediction", predict)

            #print("cross_entropy:", cro_en)
            print("accuracy:", accum)
            #print("softmax:", soft)

            x_test_batch, y_test_batch = mnist.test.next_batch(lenet5_parameter.BATCH_SIZE)
            x_test_batch = tf.reshape(x_test_batch, [lenet5_parameter.BATCH_SIZE, 28, 28, 1])
            x_test_batch = tf.pad(x_test_batch, [[0, 0], [2, 2], [2, 2], [0, 0]])
            x_test_batch = x_test_batch.eval()
            _, test_acc = sess.run([train_step, accuracy], feed_dict={xs: x_test_batch, ys: y_test_batch})
            print("test accurary:", test_acc)
            train_saver.save(sess, "./save/train_save")


    #            print("conv15:", conv15)
    #            print("fc_16:", fc16)
    # print("conv_weight5:", cw_5)
    # print("conv_weight8:", cw_8)
    # print("conv_weight15:", cw_15)
    # print("fc_17:", fc_17)

    # print("conv_weight5:", accum)


