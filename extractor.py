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
        image = tf.decode_raw(sample['image'], out_type=tf.int64)   # if not use gfile
        # image = tf.image.decode_png(sample['image'], channels=0, dtype=tf.uint8,)
        img_shape = tf.stack([sample['rows'], sample['cols'], 1], axis=0)
        label = sample['label']
        filename = sample['filename']

        return [image, label, filename, img_shape]

    def extract_image(self):
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_fn)
        dataset = dataset.shuffle(buffer_size=38916)   # need to make sure
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=self.batch_size)  # to make sure
        iterator = dataset.make_one_shot_iterator()

        return iterator

    def show_images(self):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iterator = self.extract_image()
            for i in range(self.epoch * (1080 // self.batch_size)):  # If the range is not right, it will raise OutOfRangeError

                next_image = iterator.get_next()

                img_data = sess.run(next_image)
                print(img_data[2])


if __name__ == '__main__':
    path = '/home/qinlong/PycharmProjects/NEU/Holography'
    t = TFRecordExtractor(path, './images.tfrecord', 9, 2)
    t.show_images()
