import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
# MNIST数据集相关常数
INPUT_NODE = 784  # 输入层节点,对于
OUTPUT_NODE = 10  # 输出层节点数,这个等于类别数,因为MNIST数据集需要区分的是0~9这10个数字,所以这里输出层节点数为10
# 配置神经网络的参数
LAYER1_NODE = 500  # 隐藏层节点数,这里使用只有一个隐藏层的网络结构作为样例.这个隐藏层有500个节点
BATCH_SIZE = 100  # 一个训练batch中的训练数据个数,数字越小,训练过程越接近随机梯度下降;数字越大,训练月接近梯度下降
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化在损失函数中的系数
TRAING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
'''
一个辅助函数,给定神经网络的输入和所有参数,计算神经网络的前向传播结果,在这里定义了一个使用ReLU激活函数的三层全连接神经网络。
通过加入隐藏层实现了多层网络结构,通过ReLU激活函数实现去线性化.在这个函数中也支持传入用于计算参数平均值的类,这样方便在测试时使用滑动平均模型
'''


def backward(input_tensor, avg_class, reuse=False):
    # 当没有提供滑动平均类时,直接使用参数当前取值
    if avg_class == None:
        '''
        计算输出层前向传播结果,因为在计算损失函数时就会一并计算softmax函数,
        所以这里不需要加入激活函数。而且不加入softmax不会影响预测结果。因为预测时
        使用的是不用类别对应节点输出值的相对大小,有没有softmax层对最后分类结果的计算没有影响
        于是在计算整个神经网络的前向传播时可以不加入最后的softmax层
        '''
        # 定义第一层神经网络的变量和前向传播过程
        with tf.variable_scope('layer1', reuse=reuse):
            '''
            根据传进来的reuse来判断是创建新变量还是使用自己已经创建好的.在第一次构造网络时需要创建新变量,
            以后每次调用这个函数都是直接使用reuse=True就不需要每次将变量传进来
            '''
            weights = tf.get_variable(
                name='weights',
                shape=[INPUT_NODE, LAYER1_NODE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(
                name='biases',
                shape=[LAYER1_NODE],
                initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        # 类似的定义第二层神经网络的变量和前向传播过程
        with tf.variable_scope('layer2', reuse=reuse):
            weights = tf.get_variable(
                name='weights',
                shape=[LAYER1_NODE, OUTPUT_NODE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(
                name='biases',
                shape=[OUTPUT_NODE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            layer2 = tf.matmul(layer1, weights) + biases
        # 返回最后的前向传播结果
        return layer2
    else:
        # 首先使用 avg_class.average 函数来计算得出变量的滑动平均,然后计算相应的神经网络前向传播结果
        # 定义第一层神经网络的变量和前向传播过程
        with tf.variable_scope('layer1', reuse=reuse):
            weights = tf.get_variable(
                name='weights',
                shape=[INPUT_NODE, LAYER1_NODE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(
                name='biases',
                shape=[LAYER1_NODE],
                initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(
                tf.matmul(input_tensor, avg_class.average(weights)) +
                avg_class.average(biases))
        # 类似的定义第二层神经网络的变量和前向传播过程
        with tf.variable_scope('layer2', reuse=reuse):
            weights = tf.get_variable(
                name='weights',
                shape=[LAYER1_NODE, OUTPUT_NODE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(
                name='biases',
                shape=[OUTPUT_NODE],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            layer2 = tf.matmul(
                layer1, avg_class.average(weights)) + avg_class.average(biases)
        # 返回最后的前向传播结果
        return layer2


# 训练模型的过程
def get_train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数
    weight1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weight2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    y = backward(x, None)

    # 定义存储训练轮数的变量,这个变量不需要计算滑动平均值,所以指定为不可训练的变量.
    global_step = tf.Variable(0, trainable=False)
    # 给定滑动平均衰减率和训练轮数的变量,初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均,其他辅助变量如global_step就不需要了
    # tf.trainable_variables返回的就是图上集合GraphKeys.TRAINABLE_VARIABELS中的元素
    # 这个集合的元素就是所有没有指定trainable=False的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    '''
    计算使用了滑动平均之后的前向传播结果,
    滑动平均不会改变变量本身的取值而是维护一个影子变量来记录其滑动平均值
    所以当需要使用这个滑动平均值时,需要明确调用average函数
    '''
    average_y = backward(x, variable_averages)
    '''
    计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    这里使用sparse_softmax_cross_entropy_with_logits函数来计算交叉熵,
    当分类问题只有一个正确答案时,可以使用这个函数来加速交叉熵的计算。MNIST问题的图片分类
    中只包含0～9中的一个数字,所以使用这个函数来计算交叉熵损失.这个函数的第一个参数是
    神经网络不包括softmax层的前向传播结果,第二个是训练数据的正确答案
    因为标准答案是一个长度为10的一维数组,而该函数需要提供一个正确答案的数字,所以需要
    使用tf.argmax函数来得到正确答案对应的类别编号
    '''
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算L2正则话损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失,一般只计算神经网络边上权重的正则化损失,而不使用偏置项。
    regularization = regularizer(weight1) + regularizer(weight2)
    # 总损失等于交叉熵损失和正则化损失之和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率,随着迭代的进行,更新变量时使用的学习率在这个基础上递减
        global_step,  # 当期迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY  # 学习率的衰减速度
    )
    # 使用GradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)
    '''
    在训练神经网络模型时,每过一遍数据既需要通过反向传播来更新神经网络中的参数,
    又要更新每一个参数的滑动平均值.为了一次完成多个操作,tensorflow提供了tf.control_dependencies和tf.group两种机制
    '''
    # train_op = tf.group(train_step, variables_averages_op) 与下面代码等价
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    '''
    检验使用了滑动平均模型的神经网络前向传播结果是否正确,tf.argmax(average_y,1)计算每一个样例的预测答案.其中average_y是一个
    batch_size * 10 的二维数组,每一行表示一个样例的前向传播结果.tf.argmax第二个参数1表示选取最大值的操作仅在第一个维度执行,
    即在每一行选取最大值对应的下标.于是得到的结果是一个长度为batch的数组，这个一维数组中的值表示了每一个样例对应的数字识别结果.
    tf.equal判断两个张量的每一维是否相等
    '''
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 这个运算首先将一个布尔型的数值转换为实数型,然后计算平均值.这个平均值就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据,一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的结果
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        # 准备测试数据,在真实的应用中,这部分数据在训练时是不可见的,这个数据只作为模型优劣的最后评价标准
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        # 迭代训练神经网络
        for i in range(TRAING_STEPS):
            # 每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                '''
                计算滑动平均模型在验证数据上的结果,因为MNIST数据集比较小,所以一次可以处理所有的验证数据.
                为了计算方便,本样例程序没有将验证数据划分为更小的batch.当神经网络模型比较复杂或验证数据比较大时,
                太大的batch会导致计算时间过程甚至发生内存溢出的错误.
                '''
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print(
                    "After %d training step,validation accuracy using average model is %g "
                    % (i, validate_acc))
            # 产生这一轮使用的一个batch的训练数据,并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        # 在训练结束后,在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step,test accuracy using average model is %g"
              % (TRAING_STEPS, test_acc))


# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类,这个类在初始化时会自动下载数据
    # mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    get_train(mnist)


if __name__ == "__main__":
    tf.app.run()