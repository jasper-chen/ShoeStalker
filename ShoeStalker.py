#!/usr/bin/env python

"""
October ___, A worked on camera capture things.
October 23, J+C added finding keypoints code. worked with rosbag.
October 24, C - Code runs!! HUZZAH. Doesn't do anything yet. Working on keypoints stuff. See pauls_track_object.py for possible understanding?
October 28, J - learned about implementing color histogram and SIFT.
October 28, A - added capability of reading image from Neato stream. added mouse events function. 
	Added Detect from Paul's code (altered for our use). 
November 1, A - made the code really able to read images! 
November 4, J - found the missing thing!
			J - the top right image is now static and to size
"""
import rospy
import cv2
import numpy as np
import math 

from geometry_msgs.msg import Twist, Vector3
from matplotlib import pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion, Vector3

class ShoeStalker:
	SELECTING_NEW_IMG = 0
	SELECTING_REGION_POINT_1 = 1
	SELECTING_REGION_POINT_2 = 2

	def __init__(self,descriptor):
		self.detector = cv2.FeatureDetector_create(descriptor)
		self.extractor = cv2.DescriptorExtractor_create(descriptor)
		self.matcher = cv2.BFMatcher()
		self.new_img = None
		self.new_region = None
		self.last_detection = None
		self.new_descriptors = None
		rospy.Subscriber("scan", LaserScan, self.scan_received, queue_size=1)
		self.pub=rospy.Publisher('cmd_vel',Twist,queue_size=1)
		self.new_keypoints = None
		self.magnitude=None
		self.xpos = None

		self.corner_threshold = 0.0
		self.ratio_threshold = 1.0

		self.state = ShoeStalker.SELECTING_NEW_IMG

		try:
			#for image capture 
			self.camera_listener = rospy.Subscriber("camera/image_raw", Image, self.capture)
			self.bridge = CvBridge()
		except AttributeError:
			print "ERROR!"
			pass	

	def capture(self,msg):
		# IMAGE FROM NEATO 
		#useful link for image types http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
		cv_Shoeimage = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		#Shoeimage = np.asanyarray(cv_Shoeimage)
		self.new_shoeimg = cv_Shoeimage
		if self.new_shoeimg.shape[0] == 480:
			self.image_stream = True
		else:
			self.image_stream = False
		#cv2.imshow("ShoeImage", cv_Shoeimage)
		#print "image"

		# # set up the ROI for tracking
		# region = self.new_img[self.new_region[1]:self.new_region[3],self.new_region[0]:self.new_region[2],:]
		# hsv_region =  cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
g
	def set_ratio_threshold(self,thresh):
		self.ratio_threshold = thresh

	def set_corner_threshold(self,thresh):
		self.corner_threshold = thresh

	def get_new_keypoints(self):
		""" creating keypoints for an image"""
		# #makes new image black and white
		new_img_bw = cv2.cvtColor(self.new_img,cv2.COLOR_BGR2GRAY)
		#print 'maybe?'#new_img_bw.shape
		#detect keypoints
		keyp = self.detector.detect(new_img_bw)
		#compare keypoints
		keyp = [pt
			    for pt in keyp if (pt.response > self.corner_threshold and
							   self.new_region[0] <= pt.pt[0] < self.new_region[2] and
							   self.new_region[1] <= pt.pt[1] < self.new_region[3])]
		dc, describe = self.extractor.compute(new_img_bw,keyp)
		#remap keypoints so relative to new region
		for pt in keyp:
			pt.pt = (pt.pt[0] - self.new_region[0],pt.pt[1] - self.new_region[1])
		#reassign keypoints and descriptors
		self.new_keypoints = keyp
		self.new_descriptors = describe
		#print keyp
		#print 'first keypoint point: %s' %keyp[0].pt

		#print 'keypoints, describe'
		#print keyp,describex

	def detecting(self,im):
		#print 'detecting'

		#Pauls Code - went through it and changed it to fit ours. will probably need further alterations
		img_bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		training_keypoints = self.detector.detect(img_bw)

		desc, training_descriptors = self.extractor.compute(img_bw, training_keypoints)
		#finds the k best matches for each descriptor from a query set. (http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html)
		matches = self.matcher.knnMatch(self.new_descriptors, training_descriptors, k=2)
		good_matches = []
		for m,n in matches: 
			#makes sure distance to closest match is sufficiently better than to 2nd closest
			if (m.distance < self.ratio_threshold*n.distance and
				training_keypoints[m.trainIdx].response > self.corner_threshold):
				#print 'finding matches'
				good_matches.append((m.queryIdx, m.trainIdx))

		#print 'good matches type: %s' %type(good_matches)

		#print 'good matches: %s' %good_matches

		self.matching_new_pts = np.zeros((len(good_matches),2))
		self.matching_training_pts = np.zeros((len(good_matches),2))

		track_im = np.zeros(img_bw.shape)
		for idx in range(len(good_matches)):
			match = good_matches[idx]
			self.matching_new_pts[idx,:] = self.new_keypoints[match[0]].pt
			self.matching_training_pts[idx,:] = training_keypoints[match[1]].pt
			track_im[training_keypoints[match[1]].pt[1], training_keypoints[match[1]].pt[0]] = 1.0

		#print 'matching_keypoint type: %s' %type(self.matching_new_pts)
		#print 'matching_keypoints: %s' %self.matching_new_pts

		track_im_visualize = track_im.copy()

		#converting to (x,y,width,height)
		track_region = (self.last_detection[0],self.last_detection[1],self.last_detection[2]-self.last_detection[0],self.last_detection[3]-self.last_detection[1])

		#setup criterial for termination, either 10 iteritation or move at least 1 pt
		#done to plot intermediate results of mean shift
		for max_iter in range(1,10): 
			term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, 1 )
			(ret, intermediate_region) = cv2.meanShift(track_im,track_region,term_crit)
			cv2.rectangle(track_im_visualize,(intermediate_region[0],intermediate_region[1]),(intermediate_region[0]+intermediate_region[2],intermediate_region[1]+intermediate_region[3]),max_iter/10.0,2)
		
		self.last_detection = [intermediate_region[0],intermediate_region[1],intermediate_region[0]+intermediate_region[2],intermediate_region[1]+intermediate_region[3]]

		cv2.imshow("track_win", track_im_visualize)

		#compare image of the shoe to shoe database (color histogram/SIFT technique) (this may be very time-consuming)
		#pick shoe by image of shoe with the most keypoints
		#return location of shoes (I think it might be easier to use one location of a shoe)

		#xpos = 0
		#distance = 0
		#print 'xpos,distance'
		#return xpos,distance

	def approach_shoe(self,msg):
		"""making the robot stop if it gets within a meter of the shoe (the thing directly in front of it)"""

		linear = 0
		angular = 0 
		for degree in range():
				if msg.ranges[degree] > 340.0 and msg.ranges[degree] < 20.0:
					#get position of laser points
					data_x = self.odom_pose[0] + msg.ranges[degree]*math.cos(degree*math.pi/180.0 + self.odom_pose[2])
					data_y = self.odom_pose[1] + msg.ranges[degree]*math.sin(degree*math.pi/180 + self.odom_pose[2])
					self.magnitude[degree] = math.sqrt(data_x**2 + data_y**2) 
					self.pub.publish(Twist(linear=Vector3(x=linear),angular=Vector3(z=angular)))
					

	def stalk(self):
		""" moves robot so that shoe is in center of image """
		
		#xpos,distance = self.detect(self.new_image) 
		self.xpos = (((self.last_detection[2]- self.last_detection[0])/2)+self.last_detection[0])
		
		if self.xpos == None:
			linear = 0
			angular = 0
		elif self.xpos > 155:
			linear = .05
			angular = self.xpos * -.002
		elif self.xpos < 145:
			linear = .05
			angular =(140 - self.xpos) * .002
		elif self.xpos <= 155 and self.xpos >= 145:
			linear = .1
			angular = 0
		elif self.magnitude > 1:
			#self.approach_shoe()
			linear = 0
			angular = 0
		else:
			#self.lostshoe()
			linear = 0
			angular = 0
		print linear, angular

		self.pub.publish(Twist(linear=Vector3(x=linear),angular=Vector3(z=angular)))

	def lostshoe(self):
		"""refinds a lost shoe, turn towards location of last shoe. currently just turning"""
		print 'lost shoe'

		#linear = 0
		#angular = .8
		#pub.publish(Twist(linear=Vector3(x=linear),angular=Vector3(z=angular)))


	# def run(self): #perhaps should move to if __name__ == '__main__', but this is how it is in the fixed image code
	# 	print 'run'
	# 	capture = cv2.VideoCapture(0)
	# 	ret, frame = capture.read()
	# 	cv2.namedWindow("ShoeImage")
	#	cv2.imshow("ShoeImage",frame)
	# 	cv2.setMouseCallback("ShoeImage", mouse_event)

	# def publisher(self):
	# 	rospy.init_node('ShoeStalker', anonymous = True )
	# 	pub=rospy.Publisher('cmd_vel',Twist,queue_size=10)
		
	# 	pub.publish('a')
	# 	rospy.spin()

	def mouse_event(self,event,x,y,flag,im):
		if event == cv2.EVENT_FLAG_LBUTTON:
			if self.state == self.SELECTING_NEW_IMG:
				self.new_img_visualize = frame.copy()
				self.new_img = frame
				self.new_region = None
				self.state = self.SELECTING_REGION_POINT_1
			elif self.state == self.SELECTING_REGION_POINT_1:
				self.new_region = [x,y,-1,-1]
				cv2.circle(self.new_img_visualize,(x,y),5,(255,0,0),5)
				self.state = self.SELECTING_REGION_POINT_2
			else:
				#print 'get new keypoints'
				self.new_region[2:] = [x,y]
				# print 'new region %s' %self.new_region
				self.last_detection = self.new_region
				cv2.circle(self.new_img_visualize,(x,y),5,(255,0,0),5)
				self.state = self.SELECTING_NEW_IMG
				self.get_new_keypoints()

	def set_corner_threshold_callback(self, thresh):
		""" Sets the threshold to consider an interest point a corner.  The higher the value
			the more the point must look like a corner to be considered """
		self.set_corner_threshold(thresh/1000.0)

	def set_ratio_threshold_callback(self, ratio):
		""" Sets the ratio of the nearest to the second nearest neighbor to consider the match a good one """
		self.set_ratio_threshold(ratio/100.0)

	# def teleop(self):
	# 	pub=rospy.Publisher('cmd_vel',Twist,queue_size=1)
	# 	turn_vel = .5
	# 	linear_vel = .5
	# 	r=rospy.Rate(10)

	# 	key=raw_input('drive!')
	# 	if key=='w':
	# 	    pub.publish(Twist(linear=Vector3(x=linear_vel)))
	# 	elif key=='d':
	# 	    pub.publish(Twist(angular=Vector3(z=-turn_vel)))
	# 	elif key=='s':
	# 	    pub.publish(Twist(linear=Vector3(x=-linear_vel)))
	# 	elif key=='a':
	# 	    pub.publish(Twist(angular=Vector3(z=turn_vel)))
	# 	else:
	# 		pub.publish(Twist())

	def is_in_bounding_box(self, x, y, w, h, kp):
		if kp[0] > x and kp[0] < w and kp[1] > y and kp[1] < h:
			return True
		else:
			return False

if __name__ == '__main__':
	try:
		rospy.init_node('capture', anonymous=True)
		n = ShoeStalker('SIFT')
		n.image_stream = False
		
		#create window to an interactive window to create a bounding box
		cv2.namedWindow('UI')
		cv2.createTrackbar('Corner Threshold', 'UI', 0, 100, n.set_corner_threshold_callback)
		cv2.createTrackbar('Ratio Threshold', 'UI', 100, 100, n.set_ratio_threshold_callback)
		cv2.namedWindow("ShoeImage")
		cv2.setMouseCallback("ShoeImage", n.mouse_event) #listen for mouse clicks on window

		while not(rospy.is_shutdown()):
			if n.image_stream == False:
				print 'nope'
			else:
				frame = np.array(cv2.resize(n.new_shoeimg,(n.new_shoeimg.shape[1]/2,n.new_shoeimg.shape[0]/2)))

				if n.state == n.SELECTING_NEW_IMG:
					if n.new_region != None:
						n.detecting(frame)

						# add the new image to the side
						combined_img = np.zeros((frame.shape[0],frame.shape[1]+(n.new_region[2]-n.new_region[0]),frame.shape[2]),dtype=frame.dtype)
						combined_img[:,0:frame.shape[1],:] = frame

						combined_img[0:(n.new_region[3]-n.new_region[1]),frame.shape[1]:,:] = (
								n.new_img[n.new_region[1]:n.new_region[3],
												  n.new_region[0]:n.new_region[2],:])
						# plot the matching points and correspondences
						for i in range(n.matching_new_pts.shape[0]):
							cv2.circle(combined_img,(int(n.matching_training_pts[i,0]),int(n.matching_training_pts[i,1])),2,(255,0,0),2)
							cv2.line(combined_img,(int(n.matching_training_pts[i,0]), int(n.matching_training_pts[i,1])),
												  (int(n.matching_new_pts[i,0]+frame.shape[1]),int(n.matching_new_pts[i,1])),
												  (0,255,0))

						for pt in n.new_keypoints:
							cv2.circle(combined_img,(int(pt.pt[0]+frame.shape[1]),int(pt.pt[1])),2,(255,0,0),1)
						cv2.rectangle(combined_img,(n.last_detection[0],n.last_detection[1]),(n.last_detection[2],n.last_detection[3]),(0,0,255),2)
						kp_in_box = filter(lambda x: n.is_in_bounding_box(n.last_detection[0],n.last_detection[1],n.last_detection[2],n.last_detection[3],x),n.matching_training_pts.tolist())
						cv2.imshow("ShoeImage",combined_img)

						n.stalk()
					else:
						cv2.imshow("ShoeImage",frame)
						# n.teleop()
				else:
					cv2.imshow("ShoeImage",n.new_img_visualize)
			cv2.waitKey(100)
	except rospy.ROSInterruptException: pass
