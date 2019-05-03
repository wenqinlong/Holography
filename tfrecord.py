import tensorflow as tf
import os
from PIL import Image
import numpy as np
import string


class GenerateTFRecord:
    
    def __init__(self, labels):
        self.labels = labels
 
    def convert_image_folder(self, img_folder, tfrecord_file_name):
        img_folder_paths = os.listdir(img_folder)
        img_folder_paths.sort() # 0-Z
        for folder in img_folder_paths:
            folder_path = os.path.join(img_folder, folder)  # get the path of folder which is images in
            imgs = sorted(os.listdir(folder_path))
            imgs = [os.path.abspath(os.path.join(folder_path, i)) for i in imgs]

            with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
                for img in imgs:
                    print(img)
                    example = self._convert_img(img)
                    writer.write(example.SerializeToString())

    def _convert_img(self, img_path):
        label = self._get_label_with_filename(img_path)
        img = Image.open(img_path)
        # assert img.dtype == 'uint8'

        img_int = np.array(img, dtype=np.int64)
        assert img_int.dtype == np.int64
        img_shape = img_int.shape
        assert img_shape == (100, 100)

        # convert img to string data
        img_str = img_int.tobytes()


        # get filename
        filename = os.path.basename(img_path)   # return the file name without path information

        example = tf.train.Example(features=tf.train.Features(feature={
            'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
            'rows': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[0]])),
            'cols': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[1]])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_str])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        return example

    def _get_label_with_filename(self, filename):
        basename = os.path.basename(filename).split('.')[0]
        basename = basename.split('_')[2]
        return self.labels[basename]


if  __name__ == '__main__':
    digits = list(string.digits)
    letter_uppercase = list(string.ascii_uppercase)
    keys = digits + letter_uppercase
    labels = {k: v for v, k in enumerate(keys)}
    t = GenerateTFRecord(labels)
    t.convert_image_folder('hologram_image', 'images.tfrecord')
