import tensorflow as tf

# y=ax+b
def regression(x):
    a=tf.Variable(tf.zeros([784,10]),name="a")
    b=tf.Variable()