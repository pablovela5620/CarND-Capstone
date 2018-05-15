import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from styx_msgs.msg import TrafficLight

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade tensorflow installation to v1.4.* or later!')


class TLClassifier(object):
    def __init__(self):
        # Current light variable
        self.current_light = TrafficLight.UNKNOWN
        # Current working directory path
        CWD_PATH = os.getcwd()

        # Path to frozen detection graph pb
        PATH_TO_CKPT = os.path.join(CWD_PATH, 'model', 'frozen_inference_graph.pb')

        # Loading tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)

        # Index to labels
        self.category_index = {1: {'id': 1, 'name': 'green_light'}, 2: {'id': 1, 'name': 'red_light'},
                               3: {'id': 1, 'name': 'yellow_light'}}

        # Define input and output tensors
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Expand image dimensions to have shape: [1, None, None, 3]
        image_expanded = np.expand_dims(image, axis=0)

        # Perfoming detection by running model with image_expanded
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        score_thresh = 0.65
        count = 0
        red_count = 0

        # Number of boxes
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > score_thresh:
                count += 1
                class_name = self.category_index[classes[i]['name']]

                if class_name == 'red_light':
                    red_count += 1
        if red_count < count - red_count:
            self.current_light = TrafficLight.GREEN
        else:
            self.current_light = TrafficLight.RED

        return self.current_light
