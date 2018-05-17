import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Download dataset
mnist = mnist_data.read_data_sets("MNIST_data", one_hot=True, reshape=False, validation_size=0)

# build model
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
xx = tf.reshape(x, [-1, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(xx,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
