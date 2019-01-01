import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

image_size = 28 * 28
class_num = 10
batch_size = 50
total_step = 50000
learning_rate = 0.001

x = tf.placeholder(tf.float32, [None, image_size])
y_ = tf.placeholder(tf.float32, [None, class_num])
dropout = tf.placeholder(tf.float32)

class MNIST_CNN(object):
    def __init__(self):
        def conv2d(input, w, b, name):
            return tf.nn.relu(tf.nn.conv2d(input, w, strides = [1, 1, 1, 1], padding = 'SAME') + b, name = name)


        def pool(feature, kernel_size, name):
            return tf.nn.max_pool(feature, [1, kernel_size, kernel_size, 1], [1, kernel_size, kernel_size, 1], padding = 'SAME',
                                name = name)


        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev = 0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape = shape)
            return tf.Variable(initial)

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        conv1 = conv2d(x_image, W_conv1, b_conv1, "conv1")
        pool1 = pool(conv1, 2, "pool1")

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        conv2 = conv2d(pool1, W_conv2, b_conv2, "conv2")
        pool2 = pool(conv2, 2, "pool2")

        W_fc = weight_variable([7 * 7 * 64, 1024])
        b_fc = bias_variable([1024])
        fc = tf.nn.relu(tf.matmul(tf.reshape(pool2, [-1, 7 * 7 * 64]), W_fc) + b_fc, name = "fc")
        fc_drop = tf.nn.dropout(fc, dropout)

        W_sm = weight_variable([1024, class_num])
        b_sm = bias_variable([class_num])

        y = tf.nn.softmax(tf.matmul(fc_drop, W_sm) + b_sm, name = "softmax")
        loss = -tf.reduce_sum(y_ * tf.log(y))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver=tf.train.Saver()

    def trian(self):
        for i in range(total_step):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
                train_accuracy = self.sess.run(self.accuracy, feed_dict = {x: batch_x, y_: batch_y, dropout: 1.0})
                print("step %04d, Training accuracy %.3f" % (i, train_accuracy))
            self.sess.run(self.train_step, feed_dict = {x: batch_x, y_: batch_y, dropout: 0.5})
        self.saver.save(self.sess, "models/MNIST_CNN")
        print("accuracy %.3f" % self.sess.run(self.accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, dropout: 1.0}))



if __name__ == "__main__":
    model = MNIST_CNN()
    model.trian()



