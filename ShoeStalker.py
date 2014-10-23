import cv2
import numpy as np

from matplotlib import pyplot as plt

class ShoeStalker:

	def __init__(self):

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
		cv2.destroyAllWindows() #perhaps we don't want this? 

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
	
	capture = cv2.VideoCapture(0)
	ret, frame = capture.read()
	cv2.namedWindow('image')
	cv2.imshow("image",frame)

