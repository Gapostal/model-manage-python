#coding=utf-8
from tensorflow.python.framework import graph_util
import tensorflow as tf
import os

class ModelInfo :
    pass

model_info_dictionary = {}   
    
#模型管理器初始化：默认有四个模型：mnist_simple、mnist_pro、Resnet18、Vgg16
def init_model_dictionary():
    #mnist_simple
    mnist_simple_model_info = ModelInfo()    
    mnist_simple_model_info.name = 'MNIST_SIMPLE'                                   #模型名称
    mnist_simple_model_info.pb_path = 'Model/MNIST_SIMPLE/MNIST_SIMPLE.pb'          #模型pb文件地址
    mnist_simple_model_info.first_op = 'input:0'                                    #模型第一层算子
    mnist_simple_model_info.last_op = 'softmax:0'                                   #模型最后一层算子
    mnist_simple_model_info.desc = 'Mnist simple'                                   #模型描述
    mnist_simple_model_info.accu = 0.91                                             #模型测试准确度
    
    model_info_dictionary['MNIST_SIMPLE'] = mnist_simple_model_info
    
    #mnist_pro
    mnist_pro_model_info = ModelInfo()   
    mnist_pro_model_info.name = 'MNIST_PRO'
    mnist_pro_model_info.pb_path = 'Model/MNIST_PRO/MNIST_PRO.pb'
    mnist_pro_model_info.first_op = 'input:0'
    mnist_pro_model_info.last_op = 'softmax:0'
    mnist_pro_model_info.desc = 'Mnist pro'
    mnist_pro_model_info.accu = 0.99
    
    model_info_dictionary['MNIST_PRO'] = mnist_pro_model_info
    
    #Resnet18
    resnet_18_model_info = ModelInfo()  
    resnet_18_model_info.name = 'RESNET_18'
    resnet_18_model_info.pb_path = 'Model/RESNET_18/RESNET_18.pb'
    resnet_18_model_info.first_op = 'Placeholder:0'
    resnet_18_model_info.last_op = 'prob:0'
    resnet_18_model_info.desc = 'Resnet18'
    resnet_18_model_info.accu = 0.72
    
    model_info_dictionary['RESNET_18'] = resnet_18_model_info
    
    #Vgg18
    vgg_16_model_info = ModelInfo()  
    vgg_16_model_info.name = 'VGG_16'
    vgg_16_model_info.pb_path = 'Model/VGG_16/VGG_16.pb'
    vgg_16_model_info.first_op = 'InputImage:0'
    vgg_16_model_info.last_op = 'vgg_16/fc8/squeezed:0'
    vgg_16_model_info.desc = 'Vgg16'
    vgg_16_model_info.accu = 0.72
    
    model_info_dictionary['VGG_16'] = vgg_16_model_info

#获取模型字典
def get_model_info_dictionary():    
    return model_info_dictionary
        
#增加单个模型    
def add_model_info_in_dictionary(model_name, model_pb_path, model_accu, first_op='input:0', last_op='softmax:0', model_desc=None):
    if model_info_dictionary.has_key(model_name) :
        print model_name+' is existed '
    else : 
        model_info = ModelInfo()
        model_info.name = model_name
        model_info.pb_path = model_pb_path
        model_info.first_op = first_op
        model_info.last_op = last_op
        model_info.desc = model_desc
        model_info.accu = model_accu
        
        model_info_dictionary[model_name] = model_info
        
        
#删除单个模型
def delete_model_info_in_dictionary(model_name):
    if model_info_dictionary.has_key(model_name) :
        model_info_dictionary.pop(model_name)       
    else : 
        print model_name+' is not existed '

#获取单个模型信息
def get_model_info_by_name(model_name):
    if model_info_dictionary.has_key(model_name) :
        return model_info_dictionary[model_name]       
    else : 
        print model_name+' is not existed '

        
#单个模型的修改
#当前使用“删除+新增”接口实现，先不提供单独接口

#保存模型：1）pb文件，2）保存模型信息到模型处理器中
def save_model(sess, model_name, model_accu, first_op='input:0', last_op='softmax', model_desc=None):
    if model_info_dictionary.has_key(model_name) :
        print model_name+' is existed '
    else:
        #创建保存pb文件的目录
        model_pb_dir = 'Model/'+model_name
        if not os.path.exists(model_pb_dir):
            os.mkdir(model_pb_dir)
        #保存pb文件
        model_pb_file = model_pb_dir + '/'+model_name+'.pb'
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [last_op])
        with tf.gfile.FastGFile(model_pb_file, mode='wb') as f:
            f.write(constant_graph.SerializeToString())
        #保存模型信息到模型处理器中
        add_model_info_in_dictionary(model_name, model_pb_file, model_accu, first_op, last_op, model_desc)
    