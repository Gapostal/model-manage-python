#coding=utf-8
import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.Session()

#【加载meta文件】
model_meta_path = 'Model/TEST/TEST.ckpt.meta'
model_ckpt_path = './Model/TEST/TEST.ckpt'
saver = tf.train.import_meta_graph(model_meta_path) 
saver.restore(sess, model_ckpt_path)

#【保存pb文件】
model_pb_file = 'Model/TEST/TEST.pb'
#需要知道最后一层op
last_op = 'softmax'
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [last_op])
with tf.gfile.FastGFile(model_pb_file, mode='wb') as f:
    f.write(constant_graph.SerializeToString())