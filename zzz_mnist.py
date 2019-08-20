#coding=utf-8
import input_data
import tensorflow as tf
from PIL import Image
import model_manage
from tensorflow.python.framework import graph_util
import model_manage_supplement


#【数据加载】：包括训练集与测试集
#加载Mnist训练集数据
mnist = input_data.read_data_sets("MnistData/", one_hot=True)

#【实现回归模型】
#784=28*28，表示一张图片的像素点数量
x = tf.placeholder("float", [None, 784], name='input')

#表示输入是784，输出是10（0~9）
#加上参数名称是为了在进行模型加载的时候，可以根据变量名称进行数据加载
W = tf.Variable(tf.zeros([784,10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')

#matmul+softmax
y = tf.nn.softmax(tf.matmul(x,W) + b, name='softmax')

#【训练模型】
#定义交叉熵，用来进行反向参数调节
y_ = tf.placeholder("float", [None,10])

#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化设置好了的模型
init = tf.global_variables_initializer()

#在一个Session里面启动我们的模型，并且初始化变量
sess = tf.InteractiveSession()
sess.run(init)

#训练模型，让模型循环训练1000次
for i in range(1000):
	#每一次取100张图片
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#【评估模型】
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
model_accu = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print model_accu

#【保存模型】
#1）保存成pb文件，2）加载到模型管理器里面
model_name = 'MNIST_SIMPLE'
#last_op = 'softmax'
model_manage.save_model(sess, model_name, model_accu)

# #保存成ckpt文件
# model_manage_supplement.save_model_meta(sess, 'MNIST_SIMPLE')
# #使用meta进行推理：
# mnist_dir = 'MnistData/'
# index = 2
# image = model_manage_supplement.get_one_img_from_mnist(mnist_dir, index)
# print model_manage_supplement.inference_use_meta_mode(sess, model_name, image)




