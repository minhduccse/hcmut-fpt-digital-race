#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage
import message_filters
import cv2

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from skimage.measure import find_contours

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
# Import COCO config
sys.path.append("/home/ubuntu/Workspace/Mask_RCNN/samples/coco")  # To find local version
import coco

class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "lane_segmentation"
	# number of classes (background + lane)
	NUM_CLASSES = 1 + 1

	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

class Detector:
    def __init__(self):
        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        self.config = PredictionConfig()
        self.config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)

        # Load trained weights
        self.model.load_weights('/home/ubuntu/Workspace/Weights/mask_rcnn_lane_segmentation_0009.h5', by_name=True)

        self.class_names = ['BG', 'lane']

        self.speed_pub = rospy.Publisher('Team1_speed', Float32, queue_size=10)

        self._current_image = None
        self.count = 0

        self.x_cor = None
        self.y_cor = None

    def save_coordinates(self, image, boxes, masks, class_ids, class_names):
        image = image.split("/")[-1]
        image_data = []

        for i in range(boxes.shape[0]):

            class_id = class_ids[i]
            label = class_names[class_id]

            mask = masks[:, :, i]
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                list_co_ordinates = np.moveaxis(verts, 1, 0).tolist()

                region = {"shape_attributes": {"all_points_x": list_co_ordinates[0],
                                            "all_points_y": list_co_ordinates[1]},
                        "region_attributes": {"name": {label: True}}}
                image_data.append(region)
        data = image_data
        return data

    def listener(self):
        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('listener', anonymous=True)
        imageTeam = rospy.Subscriber("Team1_image/compressed", CompressedImage, self.image_callback, queue_size = 1)

        while not rospy.is_shutdown():
            # only run if there's an image present
            if self._current_image is not None:
                results = self.model.detect([self._current_image], verbose=1)
                r = results[0]
                visualize.display_instances(self._current_image, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])
                shapes = self.save_coordinates("/home/ubuntu/catkin/lane_segmentation/validate/savedImage472.jpg", r['rois'], r['masks'], r['class_ids'], self.class_names)

                self.x_cor = None
                self.y_cor = None
                plt.figure()
                for shape in shapes:
                    if shape is not None:
                        self.x_cor = shape['shape_attributes']['all_points_x']
                        self.y_cor = shape['shape_attributes']['all_points_y']
                        plt.plot(self.x_cor+self.x_cor[:1], self.y_cor+self.y_cor[:1])
                        plt.gca().invert_yaxis()
                        plt.show()

        rospy.loginfo("Waiting for image topics...")
        rospy.spin()

    def image_callback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self._current_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        rospy.loginfo("Receiving frame...")
        cv2.imshow("Image Window", image_np)
        # img_name="savedImage{0}.jpg"
        # cv2.imwrite(img_name.format(self.count), image_np)
        # self.count = self.count + 1
        cv2.waitKey(30)
        
        self.speed_pub.publish(10.0)

if __name__ == '__main__':
    try:
        Detector().listener()
        # talker()
    except rospy.ROSInterruptException:
        pass
