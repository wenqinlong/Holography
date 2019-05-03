import tensorflow as tf
import os
import numpy as np
from PIL import Image
import matplotlib.image as mpimg


class TFRecordExtractor:
    '''
    Extract the data from TFRecord using tf.data.
    '''

    def __init__(self, folder_path, tfrecord_file, batch_size, epoch):
        self.folder_path = folder_path
        self.tfrecord_file = os.path.abspath(tfrecord_file)
        self.batch_size = batch_size
        self.epoch = epoch

    def _extract_fn(self, tfrecord):
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
            'rows': tf.FixedLenFeature([], tf.int64),
            'cols': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }

        sample = tf.parse_single_example(tfrecord, features)
        # print(sample['image'])
        # image = tf.decode_raw(sample['image'], out_type=tf.int64)   # if not use gfile
        image = tf.image.decode_png(sample['image'], channels=0, dtype=tf.uint8,)
        img_shape = tf.stack([sample['rows'], sample['cols'], 1], axis=0)
        label = sample['label']
        filename = sample['filename']
        return [image, label, filename, img_shape]

    def extract_image(self):
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_fn)
        dataset = dataset.shuffle(buffer_size=100000)   # need to make sure
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=100000)  # to make sure
        iterator = dataset.make_one_shot_iterator()
        next_image = iterator.get_next()

        return next_image

    def show_images(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            image_data = self.extract_image()
            img_data = sess.run(image_data)
            print(img_data[0], img_data[1], image_data[2], image_data[3])



if __name__ == '__main__':
    path = '/home/qinlong/PycharmProjects/NEU/Holography'
    t = TFRecordExtractor(path, './images.tfrecord', 64, 1)
    t.show_images()
