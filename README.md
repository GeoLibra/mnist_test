# mnist_test
mnist数据集的使用
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