#!/usr/bin/python3
# 两层简单神经网络(全连接)
import tensorflow as tf

# 用placeholder占位
x = tf.placeholder(tf.float32, shape=(1, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed = 1))

# 定义前前向传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 执行计算图
with tf.Session() as sess:
	# 初始化变量
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print(sess.run(y, feed_dict={x:[[0.7,0.5]]}))
