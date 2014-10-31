#!/usr/bin/env python

import numpy as np
import cv2
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

def camera(self, msg):
	 self.camera_listener = rospy.Subscriber("camera/image_raw", Image)
	 self.bridge = CvBridge()
	 cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
	 image = np.asanyarray(cv_image)
	 cv2.imshow('image', image)
	 cv2.imshow('image_raw', image_raw)

	