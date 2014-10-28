#!/usr/bin/env python

"""
October ___, A worked on camera capture things.

October 23, J+C added finding keypoints code. worked with rosbag.

October 24, C - Code runs!! HUZZAH. Doesn't do anything yet. Working on keypoints stuff. See pauls_track_object.py for possible understanding?

October 28, J - learned about implementing color histogram and SIFT.

"""
import rospy
import cv2
import numpy as np

from matplotlib import pyplot as plt
from sensor_msgs.msg import Image

class ShoeStalker:
	SELECTING_NEW_IMG = 0

	def __init__(self,descriptor):
		self.detector = cv2.FeatureDetector_create(descriptor)
		self.extractor = cv2.DescriptorExtractor_create(descriptor)
		self.matcher = cv2.BFMatcher()
		self.new_img = cv2.imread('../')
		self.new_region = None
		self.last_detection = None

		self.corner_threshold = 0.0
		self.ratio_threshold = 1.0

		self.state = ShoeStalker.SELECTING_NEW_IMG
		self.camera_listener = rospy.Subscriber("camera/image_raw", Image)
		self.bridge = CvBridge()

	def capture(self,msg):
		# for using the image from the Neato 
		#useful link for image types http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
		cv_Shoeimage = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		Shoeimage = np.array(cv_Shoeimage)

		# #To use with the webcam - for testing 
		# #take picture of shoe 
		# capture = cv.CaptureFromCAM(0)
		# img = cv.QueryFrame(capture)
		# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic') # shows image
		# #save image to specific location
		# cv.SaveImage("captured_shoe",img)
		# #read back image (necessary?)
		# img = cv2.imread('captured_shoe')
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	print 'break'
		# 	#break
		# # When everything done, release the capture
		# cap.release()

	def get_new_keypoints(self):
		#makes new image black and white
		new_imgbw = cv2.cvtColor(self.new_img,cv2.COLOR_BGR2GREY)
		#detect keypoints
		keyp = self.detector.detect(new_imgbw)
		#compare keypoints
		keyp = [point
			  for point in keyp if (point.response > self.corner_threshold and
							   self.query_region[0] <= point.point[0] < self.query_region[2] and
							   self.query_region[1] <= point.point[1] < self.query_region[3])]
		dc, describe = self.extractor.compute(new_imgbw,keyp)
		#remap keypoints so relative to new region
		for point in keyp:
			point.point = (point.point[0] - self.new_region[0],point.point[1] - self.new_region[1])
		#reassign keypoints and descriptors
		self.new_keypoints = keyp
		self.new_descriptors = describe

	def detect(self):
		print 'detect'

		#compare query of shoe to shoe references (color histogram/SIFT technique) (this may be very time-consuming)
		#pick shoe by image of shoe with the most keypoints
		#return location of shoes (I think it might be easier to use one location of a shoe)

	def stalk(self):
		print 'stalk'
		#move robot so shoe is in center of image (or will it already be like this?)
		#move towards the shoes

	def lostshoe(self):
		print 'lost shoe'
		#refind a lost shoe, turn towards location of last view

	def run(self):
		capture = cv2.VideoCapture(0)
		ret, frame = capture.read()
		cv2.namedWindow('image')
		cv2.imshow("image",frame)

if __name__ == '__main__':
	
rospy.init_node('ShoeStalker', anonymous = True )
rospy.spin()

	try:
		n = ShoeStalker('SIFT')
		n.run()
		#load image
		#show image
		#plot keypoints
	except rospy.ROSInterruptException: pass
