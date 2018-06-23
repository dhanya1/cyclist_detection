import io
import hashlib
import xml.etree.ElementTree as ET
import tensorflow as tf
import random
import os
import re
import json

from PIL import Image
from object_detection.utils import dataset_util

'''
this script automatically divides dataset into training and evaluation (10% for evaluation)
this scripts also shuffles the dataset before converting it into tfrecords
if u have different structure of dataset (rather than pascal VOC ) u need to change
the paths and names input directories(images and annotation) and output tfrecords names.
(note: this script can be enhanced to use flags instead of changing parameters on code).

default expected directories tree:
dataset- 
   -JPEGImages
   -Annotations
    dataset_to_tfrecord.py   


to run this script:
$ python dataset_to_tfrecord.py 

'''


def create_example(xml_file):
    # process the xml file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_name = root.find('imagename').text
    image_format = b'png'
    file_name = image_name.encode('utf8')
    width = int(2048)
    height = int(1024)
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    track_id = []
    unique_id = []
    for member in root.findall('children'):
        xmax.append(float(member[0][0].text)/width)
        #track_id.append(member[0][2].text.encode('utf8'))
        xmin.append(float(member[0][3].text)/ width)
        ymin.append(float(member[0][4].text)/height)
        ymax.append(float(member[0][5].text)/height)
        #unique_id.append(member[0][6].text.encode('utf8'))
        classes_text.append('cyclist'.encode('utf8'))
        classes.append(1)
        # i wrote 1 because i have only one class(cyclist)
        #  read related image
    images_dir = '/users/mscdsa2018/dsj1/PycharmProjects/action/data/tsinghuaDaimlerScripts/tar_folder/leftImg8bit/train/tsinghuaDaimlerDataset'
    full_path = os.path.join(images_dir, '{}'.format(image_name))
    print(full_path)# provide the path of images directory
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = Image.open(encoded_png_io)
    if image.format != 'PNG':
        raise ValueError('Image format not PNG')
    key = hashlib.sha256(encoded_png).hexdigest()

    # create TFRecord Example
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(file_name),
        'image/source_id': dataset_util.bytes_feature(file_name),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        #'image/children/item/trackid': dataset_util.bytes_list_feature(track_id),
        #'image/children/item/uniqueid': dataset_util.bytes_list_feature(unique_id),
    }))
    return example


def main(_):
    writer_train = tf.python_io.TFRecordWriter('train.record')
    writer_test = tf.python_io.TFRecordWriter('test.record')
    # provide the path to annotation xml files directory
    xml_files = '/users/mscdsa2018/dsj1/PycharmProjects/action/data/tsinghuaDaimlerScripts/tar_folder/labelData/train/tsinghuaDaimlerDataset/xml'+'/*.xml'
    filename_list = tf.train.match_filenames_once(xml_files)
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)
    list = sess.run(filename_list)
    random.shuffle(list)  # shuffle files list
    i = 1
    tst = 0  # to count number of images for evaluation
    trn = 0  # to count number of images for training
    for xml_file in list:
        example = create_example(xml_file)
        if (i % 10) == 0:  # each 10th file (xml and image) write it for evaluation
            writer_test.write(example.SerializeToString())
            tst = tst + 1
        else:  # the for training
            writer_train.write(example.SerializeToString())
            trn = trn + 1
        i = i + 1
        print(xml_file)
    writer_test.close()
    writer_train.close()
    print('Successfully converted dataset to TFRecord.')
    print('training dataset: # ')
    print(trn)
    print('test dataset: # ')
    print(tst)


if __name__ == '__main__':
    tf.app.run()

