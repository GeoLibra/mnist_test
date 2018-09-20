import tensorflow as tf


# y=ax+b linear model
def regression(x):
    a = tf.Variable(tf.zeros([784, 10]), name="a")
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x, a) + b)

    return y, [a, b]
