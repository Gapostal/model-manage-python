#coding=utf-8

#input_data.py的详解
#学习读取数据文件的方法，以便读取自己需要的数据库文件（二进制文件）
"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import gzip
import os
import urllib
import numpy

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  #判断目录文件是否存在，不存在则创建该目录
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  #需要读取的文件路径
  #如果文件不存在，则从指定的URL下载
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath

#用于读取二进制文件
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)

#读取图像数据，并进行维度转换
def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
	#image文件的前四位为魔术码（magic number），只有检测到这4位数据的值和2051相等时，才代表这是正确的image文件，才会继续往下读取。
	#接下来继续读取之后的4位，代表着image文件中，所包含的图片的数量（num_images）。
	#再接着读4位，为每一幅图片的行数（rows），再后4位，为每一幅图片的列数（cols）。
	#最后再读接下来的rows * cols * num_images位，即为所有图片的像素值。
	#最后再将读取到的所有像素值装换为[index, rows, cols, depth]的4D矩阵。这样就将全部的image数据读取了出来。
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
	#四个字节一读，从而获取图片的N(num),C(1),H(rows),W(cols)
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
	#读取像素值数据
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
	#将像素值数据转换为[num_images, rows, cols, 1]的4D矩阵
    data = data.reshape(num_images, rows, cols, 1)
    return data

#将稠密标签向量变成稀疏的标签矩阵
#eg：若原向量的第i行为3，则对应稀疏矩阵的第i行下标为3的值为1，其余为0
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  #labels_dense.ravel()将整个数组展成一个一维数组
  #labels_dense.flat[i]即将labels_dense看成一个一维数组，取其第i个变量
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

#读取标签数据，并将数据转换为稀疏的标签矩阵
#具体同extract_images
def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels

#设置数据的Set[image, label]，以及一些参数的初始化
class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)

      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]  #申请一片连续的为1.0的空间，空间大小为784，数据类型是浮点型
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    #若当前训练读取的index>总体的images数时，则读取开始的batch_size大小的数据
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000
  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)
  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)
  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)
  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)
  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]
  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)
  return data_sets
