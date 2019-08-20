本工具意在实现基于python的TensorFlow已有模型的管理，包括模型增加，删除，使用，保存（pb以及ckpt格式）
当前支持的模型有：
MNIST_SIMPLE：用来进行手写数字识别，网络训练精度：91%
MNIST_PRO：用来进行手写数字识别，网络训练精度：99%
RESNET_18：用来进行ImageNet数据集中物体识别。（当前识别准确度较低，还需要进行分析）
VGG_16：用来进行ImageNet数据集中物体识别。

其中，MNIST_SIMPLE和MNIST_PRO提供数据集和训练源码，可以进行模型训练；RESNET_18和VGG_16只提供了pb文件。

【限制】：
1、需要安装python2.7
2、需要安装TensorFlow1.4
3、需要安装TensorBoard
4、需要安装相关的依赖库

【脚本文件说明】：
1、change_to_image.py：
可以将Mnist数据集数据从t10k-images.idx3-ubyte转换为一张张图片，脚本当前设置的图片数量是：100

2、get_log_from_pb.py：
从pb文件生成logs，从而能够调用TensorBoard查看网络，具体参考：http://rnd-github.huawei.com/zhangxiaochi/Kirin-AI-SW-VII/issues/31

3、get_pb_from_meta.py：
从meta和chpt文件生成pb模型文件

4、imagenet_classes.py：
ImageNet数据集的分类类型文件

5、input_data.py：
Mnist数据集的下载以及读入成流的文件

6、model_manage.py：
模型管理器文件，主要包括模型的：初始化，增，删，查

7、model_manage_supplement.py：
模型管理器补充的功能文件，提供了一些额外的功能

8、use_model.py：
主要使用的脚本，其中只需要设置：model_name和img_dir,就可以调用已有的模型进行图片处理

9、zzz_mnist.py：
使用简单的方法进行mnist手写数字识别的脚本，包括训练和模型保存

10、zzz_mnist_pro.py：
使用较复杂的方法进行mnist手写数字识别的脚本，包括训练和模型保存

【使用】：
1、将pb文件保存到指定的目录，参考《model_manage.py》脚本中的地址

2、配置《use_model.py》中model_name和img_dir，使用：python use_model.py可以查看输出结果

（当前MNIST_SIMPLE、MNIST_PRO、VGG_16网络都可以使用，RESNET_18网络准确度问题待分析）