# mnist_test
mnist数据集的使用
## one_hot
```python
data = input_data.read_data_sets('MNIST_data', one_hot=True)
```
one_hot=True表示将样本标签转化为one_hot编码。  
举例：假如一共10类。  
0-->1000000000  
1-->0100000000  
2-->0010000000
3-->0001000000  
以此类推。只有一位为1，1所在的位置就代表着第几类。
## 保存模型
```python
saver=tf.train.Save() # 实例化saver对象
with tf.Session() as sess: # 在with结构for循环中一定轮数时 保存模型到当前回话
    for i in range(STEPS):
        if i % 轮数 ==0: 
            saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
```
## 加载模型
```python
with tf.Session() as sess:
    ckpt=tf.train.get_checkpoint_state(存储路径)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
```
## 实例化可还原滑动平均值的saver
```python
ema=tf.train.ExponentialMovineAverage(滑动平均基数)
ema_restore=ema.variables_to_restore()
saver=tf.train.Saver(ema_restore)
```
## 准确率计算方法
```python
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
```
## 遇到的问题  
1. CUDA_ERROE_OUT_OF_MEMORY  
服务器的GPU大小为M,tensorflow只能申请N（N<M）,也就是tensorflow告诉你 不能申请到GPU的全部资源,然后就不干了  
解决方案:
```python
config = tf.ConfigProto(allow_soft_placement=True)
# 最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# 开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
```
2. Attempting to use uninitialized value  
解决方法:变量初始化  
```python
sess.run(tf.global_variables_initializer())
```
3. Attempting to use uninitialized value Variable Caused by op u'Variable/read'  
解决方法:
将 `sess.run(tf.global_variables_initializer())` 放到后面
