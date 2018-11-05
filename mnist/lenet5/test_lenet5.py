import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward_lenet5
import backforward_lenet5
import numpy as np
TEST_INTERVAL_SECS = 5


def test(mnist):
    # 创建一个默认图,在该图中执行以下操作(多数操作和train中一样)
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            mnist.test.num_examples, forward_lenet5.IMAGE_SIZE,
            forward_lenet5.IMAGE_SIZE, forward_lenet5.NUM_CHANNELS
        ])
        y_ = tf.placeholder(tf.float32, [None, forward_lenet5.OUTPUT_NODE])
        y = forward_lenet5.forward(x, False, None)
        ema = tf.train.ExponentialMovingAverage(
            backforward_lenet5.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        # 判断预测值和实际值是否相同
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 求平均得到准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                    backforward_lenet5.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 根据读入的模型名字切分出该模型是属于迭代了多少次保存的
                    global_step = ckpt.model_checkpoint_path.split(
                        '/')[-1].split('-')[-1]
                    reshaped_x = np.reshape(
                        mnist.test.images,
                        (mnist.test.num_examples, forward_lenet5.IMAGE_SIZE,
                         forward_lenet5.IMAGE_SIZE,
                         forward_lenet5.NUM_CHANNELS))
                    # 计算出测试集上准确率
                    accuracy_score = sess.run(
                        accuracy,
                        feed_dict={
                            x: reshaped_x,
                            y_: mnist.test.labels
                        })
                    print("After %s training step,test accuracy = %g" %
                          (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
                time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets("../data", one_hot=True)
    test(mnist)


if __name__ == "__main__":
    main()
