import tensorflow as tf
import os
import numpy as np

class TFRecordExtractor:
    '''
    Extract the data from TFRecord using tf.data.
    '''

    def __init__(self, tfrecord_file, batch_size, epoch):
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

        sample = tf.parse_simple_example(tfrecord, features)

        image = tf.image.decode_png(sample['image'])
        img_shape = tf.stack([sample['rows'], sample['cols']])
        label = sample['label']
        filename = sample['filename']
        return [image, label, filename, img_shape]

    def extract_image(self):
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = dataset.map(self._extract_fn)
        dataset = dataset.shuffle(buffer_size=None)   # need to make sure
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=)  # to make sure
        iterator = dataset.make_one_shot_iterator()
        next_image = iterator.get_next()

        return next_image