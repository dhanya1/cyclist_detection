import tensorflow as tf
import numpy as np
from PIL import Image


class CycistDetector(object):
    def __init__(self):
        PATH_TO_MODEL = '/users/mscdsa2018/dsj1/fast_rcnn/lib/python3.5/site-packages/tensorflow/models/research/object_detection/custom_scripts/models/model/fine_tuned_model/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
            print(boxes)
        return boxes, scores, classes, num

s = CycistDetector()
img = Image.open('/users/mscdsa2018/dsj1/PycharmProjects/action/data/tsinghuaDaimlerScripts/tar_folder/leftImg8bit/train/tsinghuaDaimlerDataset/tsinghuaDaimlerDataset_2014-11-20_075523_000002251_leftImg8bit.png')
s.get_classification(img)
