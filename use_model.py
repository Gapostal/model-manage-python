#coding=utf-8
import input_data
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import model_manage
from imagenet_classes import class_names

#读取指定位置的单张图片，转换成指定维度[1, 784]的tensor
def get_one_img_mnist(img_dir):
    image = Image.open(img_dir)
    image = image.resize([28, 28])
    image = np.array(image)
    # cast(x, dtype, name=None) 将x的数据格式转化成dtype
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [1, 784])
    return image

#获取手写数字识别的输出，结果为概率最大数字
def get_mnist_out(img_out_softmax, axis=1):
    prediction_labels = np.argmax(img_out_softmax, axis)
    print "label:",prediction_labels
    return prediction_labels
    
#读取指定位置的单张图片，转换成指定维度[1, 224, 224, 3]的tensor
def get_one_img_resnet(img_dir):
    image = Image.open(img_dir)
    image = image.resize([224, 224])
    image = np.array(image)
    # cast(x, dtype, name=None) 将x的数据格式转化成dtype
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [1, 224, 224, 3])
    return image   
 
#获取物体识别的输出，结果为概率最大的前5个物体
def get_resnet_out(img_out_softmax):
    # print 'max is : ' + bytes(np.argmax(img_out_softmax, 0))
    # print 'min is : ' + bytes(np.argmin(img_out_softmax, 0))
    # print np.sum(img_out_softmax)
    preds = (np.argsort(img_out_softmax)[::-1])[0:5]
    for p in preds:
        # 打印结果：概率，标签，类型
        print(img_out_softmax[p], p, class_names[p])
    return preds

#使用pb文件进行推理，输入为img_path和model_name        
def inference_use_mode(model_name, img_path):
    pb_file_path = model_manage.get_model_info_dictionary().get(model_name).pb_path
    
    #定义graphDef用来保存解析后的网络graph
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            #获取首层输入节点
            input_first_op = sess.graph.get_tensor_by_name(model_manage.get_model_info_by_name(model_name).first_op)
            # print input_first_op

            #获取最后一层输出节点
            out_last_op = sess.graph.get_tensor_by_name(model_manage.get_model_info_by_name(model_name).last_op)
            # print out_last_op

            if model_name == 'MNIST_SIMPLE':
                #其中eval把tensor转换为numpy
                img_out_softmax = sess.run(out_last_op, feed_dict={input_first_op:get_one_img_mnist(img_path).eval(session=sess)})
                get_mnist_out(img_out_softmax)
            elif model_name == 'MNIST_PRO':
                #对于MNIST_PRO模型，需要增加keep_prob:1.0
                keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
                img_out_softmax = sess.run(out_last_op, feed_dict={input_first_op:get_one_img_mnist(img_path).eval(session=sess), keep_prob:1.0})
                get_mnist_out(img_out_softmax)
            elif model_name == 'RESNET_18':
                #Resnet18最后一层为softmax层，所以不要增加一层softmax层
                img_out_softmax = sess.run(out_last_op, feed_dict={input_first_op:get_one_img_resnet(img_path).eval(session=sess)})[0]
                get_resnet_out(img_out_softmax)
            elif model_name == 'VGG_16':
                #Resnet18最后一层为fc层，所以要增加一层softmax层
                softmax_tmp = tf.nn.softmax(out_last_op, name='softmax_tmp')
                img_out_softmax = sess.run(softmax_tmp, feed_dict={input_first_op:get_one_img_resnet(img_path).eval(session=sess)})[0]
                get_resnet_out(img_out_softmax)
            else:
                print model_name+' is not existed'

        
#【main部分】：
#模型管家初始化
model_manage.init_model_dictionary()

#使用的模型名称
#model_name = 'MNIST_SIMPLE'
#model_name = 'MNIST_PRO'
model_name = 'RESNET_18'
#model_name = 'VGG_16'

#待推理图片
#img_dir = 'test_data/mnist_data/2018-4-20-2.gif'
#img_dir = 'test_data/mnist_data/99.png'
#img_dir = 'test_data/resnet_data/n01440764_39.jpg'
img_dir = 'test_data/resnet_data/ILSVRC2012_val_00000002.JPEG'

#【结果输出】
inference_use_mode(model_name, img_dir)
#print model_manage.get_model_info_by_name(model_name)


