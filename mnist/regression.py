import input_data
import tensorflow as tf
import module
import os
data = input_data.read_data_sets('MNIST_data', one_hot=True)

# create model
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = module.regression(x)

# train
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
learning_rate = 0.001
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 断点续训
    ckpt = tf.train.get_checkpoint_state(
        os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for item in range(50000):
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print((sess.run(
        accuracy, feed_dict={
            x: data.test.images,
            y_: data.test.labels
        })))
    path = saver.save(
        sess,
        os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False,
        write_state=False)
    print("Saved:", path)