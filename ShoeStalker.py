"""
October ___, A worked on camera capture things.

October 23, J+C 
"""
import cv2
import numpy as np

from matplotlib import pyplot as plt

class ShoeStalker:
	SELECTING_NEW_IMG = 0

	def __init__(self,descriptor):
		self.detector = cv2.FeatureDetector_create(descriptor)
		self.extractor = cv2.DescriptorExtractor_create(descriptor)
		self.matcher = cv2.BFMatcher()
		self.new_img = None
		self.new_region = None
		self.last_detection = None

		self.corner_threshold = 0.0
		self.ratio_threshold = 1.0

		self.state = ShoeStalker.SELECTING_NEW_IMG

	def capture(self):
		#take picture of shoe 
		capture = cv.CaptureFromCAM(0)
		img = cv.QueryFrame(capture)
		plt.imshow(img, cmap = 'gray', interpolation = 'bicubic') # shows image
		#save image to specific location
		cv.SaveImage("captured_shoe",img)
		#read back image (necessary?)
		img = cv2.imread('captured_shoe')
		    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
		# When everything done, release the capture
		cap.release()
		#cv2.destroyAllWindows() #perhaps we don't want this? Don't want this till the end

	def get_keypoints(self):
		#makes new image black and white
		new_imgbw = cv2.cvtColor(self.new_img,cv2.COLOR_BGR2GREY)
		keyp = self.detector.detect(new_imgbw)
		keyp = [pt
			  for pt in kp if (pt.response > self.corner_threshold and
							   self.query_region[0] <= pt.pt[0] < self.query_region[2] and
							   self.query_region[1] <= pt.pt[1] < self.query_region[3])]
		dc, des = self.extractor.compute(new_imgbw,keyp)
		#remap so relative to region
		for point in keyp:
			point.point = (point.point[0] - self.new_region[0],point.point[1] - self.new_region[1])

	def detect(self):
		#find the shoe

		#compare keypoints/color to shoe database
		#return location of shoes

	def stalk(self):
		#move robot so shoe is in center of image (or will it already be like this?)
		#move towards the shoes

	def lostshoe():
		#refind a lost shoe, turn towards location of last view

if __name__ == '__main__':
	tracker = ShoeStalker('SIFT')

	capture = cv2.VideoCapture(0)
	ret, frame = capture.read()
	cv2.namedWindow('image')
	cv2.imshow("image",frame)

