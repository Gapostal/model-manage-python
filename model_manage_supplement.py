#coding=utf-8
import tensorflow as tf
import os
import input_data

#加载meta模型，返回graph
def get_model_graph(sess, model_meta_path, model_ckpt_path):
	saver = tf.train.import_meta_graph(model_meta_path) 
	saver.restore(sess, model_ckpt_path)
	#获取已训练生成模型的graph
	return tf.get_default_graph()

#从mnist测试集中读取单张图片，并转换为(1,784)维度的tensor
def get_one_img_from_mnist(mnist_dir, index):
    mnist = input_data.read_data_sets(mnist_dir, one_hot=True)	
    image = mnist.test.images[index].reshape(1,784)
    return image
    
#调用save接口保存模型文件为ckpt和meta文件
def save_model_meta(sess, model_name):
    model_pb_dir = 'Model/'+model_name
    if not os.path.exists(model_pb_dir):
        os.mkdir(model_pb_dir)
    model_ckpt_file = model_pb_dir + '/'+model_name+'.ckpt'
    saver = tf.train.Saver()
    saver.save(sess, model_ckpt_file)

#从meta文件中获取模型的graph    
def get_model_from_meta(sess, model_name):
    model_meta_path = 'Model/'+model_name+'/'+model_name+'.ckpt.meta'
    model_ckpt_path = './Model/'+model_name+'/'+model_name+'.ckpt'
    graph = get_model_graph(sess, model_meta_path, model_ckpt_path)
    return graph
    
#MNIST_SIMPLE模型的推理函数
def inference_use_mnist_simple_model(sess, graph, image):
    #获取待使用的参数
    W = graph.get_tensor_by_name('W:0') 
    b = graph.get_tensor_by_name('b:0')
    
    result = tf.nn.softmax(tf.matmul(image,W) + b)
    return sess.run(tf.argmax(result,1))
    
#根据model_name调用对应的模型进行数据推理（待扩展）
def inference_use_meta_mode(sess, model_name, image):        
    if model_name == 'MNIST_SIMPLE' :
        graph = get_model_from_meta(sess, model_name)

        #获取结果
        return inference_use_mnist_simple_model(sess, graph, image)
    else :
        return -1