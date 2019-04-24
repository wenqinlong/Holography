import tensorflow as tf
import os
import matplotlib.image as mpl_img

class GenerateTFRecord:
    
    def __init__(self, labels):
        self.labels = labels
 
    def convert_image_folder(self, img_folder, tfrecord_file_name):
        img_folder_path = os.list(img_folder)
        img_paths = [os.path.abspath(os.path.join(img_folder, i)) for i in img_folder_path]

        with tf.python.io.TFRecordWriter(tfrecord_file_name) as writer:
            for img_path in img_paths:
                example = self._convert_im(img_path)
                writer.write(example.SerializeToString())

    def _convert_img(self, img_path):
        label = self._get_label_with_file_name(img_path)
        img_data = mpl_img.imread(img_path)
        # convert img to string data
        img_str = img_data.tostring()
        img_shape = img_data.shape
        # get filename
        filename = os.path.basename(img_data)   # return the file name without path information

        example = tf.train.Example(features = tf.train.Features(feature = {

        }))




    
    
