#!/usr/bin/env python

"""
October ___, A worked on camera capture things.

October 23, J+C added finding keypoints code. worked with rosbag.

October 24, C - Code runs!! HUZZAH. Doesn't do anything yet. Working on keypoints stuff. See pauls_track_object.py for possible understanding?

October 28, J - learned about implementing color histogram and SIFT.

October 28, A - implementing subscribers for image from Neato. added ability to 
				choose region in image 

"""
import rospy
import cv2
import numpy as np

from geometry_msgs.msg import Twist, Vector3
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from std_msgs.msg import String

class ShoeStalker:
	SELECTING_NEW_IMG = 0
	SELECTING_SHOE_PT_1 = 1
	SELECTING_SHOE_PT_2 = 2

	def __init__(self,descriptor):

		self.detector = cv2.FeatureDetector_create(descriptor)
		self.extractor = cv2.DescriptorExtractor_create(descriptor)
		self.matcher = cv2.BFMatcher()
		self.new_img = cv2.imread('./shoefront/frame0000.jpg')
		self.new_region = None
		self.last_detection = None

		self.corner_threshold = 0.0
		self.ratio_threshold = 1.0
		self.state = ShoeStalker.SELECTING_NEW_IMG

		try:
		#for image capture 
		self.camera_listener = rospy.Subscriber("camera/image_raw", Image)
		self.bridge = CvBridge()
		except AttributeError:
			pass
		
		self.new_img = None
		self.new_region = None
		self.last_detection = None


	def capture(self,msg):
		# for using the image from the Neato 
		#useful link for image types http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
		cv_Shoeimage = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		Shoeimage = np.array(cv_Shoeimage)

		# set up the ROI for tracking
		region = self.new_img[self.new_region[1]:self.new_region[3],self.new_region[0]:self.new_region[2],:]
		hsv_region =  cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

		# #To use with the webcam - for testing 
		# #take picture of shoe 
		# capture = cv.CaptureFromCAM(0)
		# img = cv.NewFrame(capture)
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
							   self.new_region[0] <= point.point[0] < self.new_region[2] and
							   self.new_region[1] <= point.point[1] < self.new_region[3])]
		dc, describe = self.extractor.compute(new_imgbw,keyp)
		#remap keypoints so relative to new region
		for point in keyp:
			point.point = (point.point[0] - self.new_region[0],point.point[1] - self.new_region[1])
		#reassign keypoints and descriptors
		self.new_keypoints = keyp
		self.new_descriptors = describe

	def detect(self, new_keypoints, new_descriptors):
		print 'detect'

		

		#compare image of the shoe to shoe database (color histogram/SIFT technique) (this may be very time-consuming)
		#pick shoe by image of shoe with the most keypoints
		#return location of shoes (I think it might be easier to use one location of a shoe)

	def stalk(self,xpos): #potentially add distance to shoe if that happens
		print 'stalk'
		#move robot so shoe is in center of image (or will it already be like this?)
		#move towards the shoes

		#if xpos > 0:
			#linear = .5
			#angular = xpos * something depending on what the units of xpos are
			#pub.publish(Twist(linear=Vector3(x=linear),angular=Vector3(z=angular)))
		#else:
			#self.lostshoe()

	def lostshoe(self):
		print 'lost shoe'
		#refind a lost shoe, turn towards location of last view
		#currently just turning

		#linear = 0
		#angular = 0.4
		#pub.publish(Twist(linear=Vector3(x=linear),angular=Vector3(z=angular)))

	def run(self):
		print 'run'
		capture = cv2.VideoCapture(0)
		ret, frame = capture.read()
		cv2.namedWindow('image')
		cv2.imshow("image",frame)

	def image(self):
		print 'image'
		frame = self.new_img
		cv2.namedWindow('image')
		cv2.imshow("image",frame)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def publisher(self):
		pub=rospy.Publisher('cmd_vel',Twist,queue_size=10)
		rospy.init_node('ShoeStalker', anonymous = True )
		pub.publish('a')
		rospy.spin()

	def mouse_event(event, x):
		if event == cv2.EVENT_FLAG_LBUTTON:
			if tracker.state == tracker.SELECTING_NEW_IMG:
				tracker.new_img_visualize = frame.copy()
				tracker.new_img = frame
				tracker.new_region = None
				tracker.state = tracker.SELECTING_REGION_PT_1
			elif tracker.state == tracker.SELECTING_REGION_PT_1:
				tracker.new_region = [x,y,-1,-1]
				cv2.circle(tracker.new_img_visualize,(x,y),5,(255,0,0),5)
				tracker.state = tracker.SELECTING_ROI_PT_2
			else:
				tracker.new_region[2:] = [x,y]
				tracker.last_detection = tracker.new_region
				cv2.circle(tracker.new_img_visualize,(x,y),5,(255,0,0),5)
				tracker.state = tracker.SELECTING_NEW_IMG
				tracker.get_new_keypoints()		

if __name__ == '__main__':
	try:
		n = ShoeStalker('SIFT')
		n.image()
		n.publisher()
		
		#load image
		#show image
		#plot keypoints
	except rospy.ROSInterruptException: pass
