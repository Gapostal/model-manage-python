#coding=utf-8
import input_data
import tensorflow as tf
import model_manage
from tensorflow.python.framework import graph_util

#【权值初始化】
#定义初始化权重函数
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#定义初始化偏置函数
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#【卷积和池化】
#定义conv算子，使用1步长（stride size），0边距（padding size）的模板
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#定义pooling算子，使用2x2大小的模板做max pooling  
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#【数据加载】：包括训练集与测试集
#加载Mnist训练集数据
mnist = input_data.read_data_sets("MnistData/", one_hot=True)

#【运行TensorFlow的InteractiveSession】
sess = tf.InteractiveSession()

#【占位符】
#我们通过为输入图像和目标输出类别创建节点，来开始构建计算图。
x = tf.placeholder("float", shape=[None, 784], name='input')
y_ = tf.placeholder("float", shape=[None, 10])
						
#【第一层卷积】
#它由一个卷积接一个max pooling完成。
#卷积在每个5x5的patch中算出32个特征。卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 
#而对于每一个输出通道都有一个对应的偏置量。
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
x_image = tf.reshape(x, [-1,28,28,1])

#把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#【第二层卷积】
#第二层中，每个5x5的patch会得到64个特征。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
#我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#为了减少过拟合，我们在输出层之前加入dropout。
#我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
#这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 
#TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
#所以用dropout的时候可以不用考虑scale。
keep_prob = tf.placeholder("float", name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#【输出层】
#添加一个softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='softmax')

#【训练和评估模型】
#为了进行训练和评估，我们使用与之前简单的单层SoftMax神经网络模型几乎相同的一套代码，
#只是我们会用更加复杂的ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例。
#然后每100次迭代输出一次日志。
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print "step %d, training accuracy %g"%(i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

model_accu = accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print model_accu
	
#【保存模型】
#1）pb文件，2）加载到模型管理器里面
model_name = 'MNIST_PRO'
#last_op = 'softmax'
model_manage.save_model(sess, model_name, model_accu)

