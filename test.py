import tensorflow as tf
from lenet import lenet5_parameter
from tensorflow.examples.tutorials.mnist import input_data
from lenet import lenet5
import time

mnist_path = "mnist/"
#mnist_path = "oss://hualangdeeplearning.oss-cn-shanghai-internal.aliyuncs.com/cat_and_dog/mnist/"

mnist = input_data.read_data_sets(mnist_path, one_hot=True)
global_step = tf.Variable(0)
xs = tf.placeholder(tf.float32, lenet5_parameter.TRAIN_SIZE)
ys = tf.placeholder(tf.float32, lenet5_parameter.OUTPUT_SIZE)

prediction = lenet5.lenet5_model(xs)
soft_max_result = tf.nn.softmax(prediction)
# print(prediction)

#loss = cross_entropy + weight_loss

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys, 1), tf.argmax(soft_max_result, 1)), tf.float32))

decayed_learning_rate = tf.train.exponential_decay(1e-4, global_step = global_step, decay_steps=200, decay_rate=0.96)


init = tf.global_variables_initializer()
test_saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    test_saver.restore(sess=sess, save_path="./save/train_save")
    x_test_batch, y_test_batch = mnist.test.next_batch(lenet5_parameter.BATCH_SIZE)
    x_test_batch = tf.reshape(x_test_batch, [lenet5_parameter.BATCH_SIZE, 28, 28, 1])
    x_test_batch = tf.pad(x_test_batch, [[0, 0], [2, 2], [2, 2], [0, 0]])
    x_test_batch = x_test_batch.eval()
    test_acc = sess.run([ accuracy], feed_dict={xs: x_test_batch, ys: y_test_batch})
    print("test accurary:", test_acc)



