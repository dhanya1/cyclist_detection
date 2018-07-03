import io
import hashlib
import tensorflow as tf
import random
import os
import json
from PIL import Image
from object_detection.utils import dataset_util

'''
This scripts shuffles the dataset before converting it into tfrecords.

default expected directories tree:
dataset- 
   -PNGImages
   -JSON Annotations
  image_to_tfrecord.py   


to run this script:
$ python image_to_tfrecord.py -a train (for training)
$ python image_to_tfrecord.py -a test (for training)

'''

class CreateTFRecord():
    all_records = {}
    def __init__(self):
        self.width = int(2048)
        self.height = int(1024)
        self.xmin = []
        self.ymin = []
        self.xmax = []
        self.ymax = []
        self.classes = []
        self.classes_text = []
        self.image_format = b'png'

    def __reading_json(self, json_file):
        with open(json_file, 'r') as file:
            json_content = file.read()
        self.json_Dict = json.loads(json_content)
        image_name = self.json_Dict['imagename']
        CreateTFRecord.all_records[image_name] = self


    def create_records(self, images_dir, json_dir):
        '''
        :param image_dir: Give full path of the directory that has test images
        :param json_dir:  Give full path of the directory that has test annotations
        :return:
        '''
        self.__reading_json(json_dir)
        self.image_name = self.json_Dict['imagename']
        self.full_path = os.path.join(images_dir, '{}'.format(self.image_name))
        list_of_objects = self.json_Dict['children']
        for objects in list_of_objects:
            self.xmax.append(float(objects['maxcol'])/self.width)
            self.xmin.append(float(objects['mincol'])/ self.width)
            self.ymin.append(float(objects['minrow'])/self.height)
            self.ymax.append(float(objects['maxrow'])/self.height)
            self.classes_text.append(objects['identity'].encode('utf8'))
            self.classes.append(1)
        example = self.__add_features()
        return example

    def __add_features(self):
        # create TFRecord Example
        with tf.gfile.GFile(self.full_path, 'rb') as fid:
            encoded_png = fid.read()
        encoded_png_io = io.BytesIO(encoded_png)
        image = Image.open(encoded_png_io)
        if image.format != 'PNG':
            raise ValueError('Image format not PNG')
        key = hashlib.sha256(encoded_png).hexdigest()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(self.height),
            'image/width': dataset_util.int64_feature(self.width),
            'image/filename': dataset_util.bytes_feature(self.image_name.encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(self.image_name.encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature(self.image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(self.xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(self.xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(self.ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(self.ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(self.classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(self.classes),
            # 'image/children/item/trackid': dataset_util.bytes_list_feature(track_id),
            # 'image/children/item/uniqueid': dataset_util.bytes_list_feature(unique_id),
        }))
        return example

def writing_train():
    writer_train = tf.python_io.TFRecordWriter('train.record')
    json_files = '/home/tar_folder/labelData/train/tsinghuaDaimlerDataset/*.json'
    images_dir = '/home/tar_folder/labelData/train/leftImg8bit/train/tsinghuaDaimlerDataset'
    filename_list = tf.train.match_filenames_once(json_files)
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)
    list = sess.run(filename_list)
    random.shuffle(list)  # shuffle files list
    for json_file in list:
        obj = CreateTFRecord(images_dir, json_file)
        example = obj.create_records(json_file)
        writer_train = tf.python_io.TFRecordWriter('train.record')
        writer_train.write(example.SerializeToString())
    writer_train.close()


def writing_test():
    # provide the path to annotation xml files directory
    json_files = '/home/tar_folder/labelData/test/tsinghuaDaimlerDataset/*.json'
    images_dir = '/home/tar_folder/leftImg8bit/test/tsinghuaDaimlerDataset'
    filename_list = tf.train.match_filenames_once(json_files)
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)
    list = sess.run(filename_list)
    random.shuffle(list)  # shuffle files list
    for json_file in list:
        obj = CreateTFRecord()
        example = obj.create_records(images_dir, json_file)
        writer_test = tf.python_io.TFRecordWriter('test.record')
        writer_test.write(example.SerializeToString())
    writer_test.close()

def main(arguments):
    action = input('Enter test or train: ').strip()
    if action.lower() == 'train':
        writing_train()
    elif action.lower() == 'test':
        writing_test()
    else:
        print('Invalid input. Enter test or train')

if __name__ == '__main__':
    tf.app.run()
    print('Success')
