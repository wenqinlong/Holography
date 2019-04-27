import tensorflow as tf
import os
from PIL import Image
import numpy as np
import string


class GenerateTFRecord:
    '''
    Convert the image to binary records and store them in TFRecord format.
    It is efficient to read data.
    '''

    def __init__(self, labels):
        self.labels = labels

    def _convert_image_folder(self, img_folder, tfreocrd_file_name):
        img_folder_paths = os.listdir(img_folder)
        img_folder_paths.sort()
        for folder in img_folder_paths:
            folder_path = os.path.join(img_folder, folder)
            imgs = sorted(os.listdir(folder_path))
            imgs = [os.path.abspath(os.path.join(folder_path, i)) for i in imgs]  # get the absolute path of every image

            with tf.python_io.TFRecordWriter(tfreocrd_file_name) as writer:
                for img in imgs:
                    example = self._convert_image(img)
                    writer.write(example.SerializeToString())

    def _convert_image(self, img_path):
        label = self._get_label_with_filename(img_path)
        img = Image.open(img_path)

        img_int = np.array(img, dtype=np.int64)
        img_shape = img_int.shape
        assert img_shape == (100, 100)

        filename = os.path.basename(img_path)

        # Read image data in terms of bytes
        with tf.gfile.GFile(img_path, 'rb') as fid:
            image_data = fid.read()

        example = tf.train.Example(features=tf.train.Features(feature={
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),   # if not encode, raise TypeError: '0000_0_0_size_30_angle_0.png' has type str, but expected one of: bytes
            'rows': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[0]])),
            'cols': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[1]])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        return example

    def _get_label_with_filename(self, filename):
        basename = os.path.basename(filename).split('.')[0]
        basename = basename.split('_')[2]
        return self.labels[basename]


if __name__ == '__main__':
    digits = list(string.digits)
    letter_uppercase = list(string.ascii_uppercase)
    keys = digits + letter_uppercase
    labels = {k: v for v, k in enumerate(keys)}
    t = GenerateTFRecord(labels)
    t._convert_image_folder('hologram_image', 'images_gfile.tfrecord')
