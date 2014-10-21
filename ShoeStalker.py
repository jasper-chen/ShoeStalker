import cv2
import numpy as np

class ShoeStalker:

	def __init__(self):

	def detect(self):
		#find the shoe

		#compare keypoints/color to shoe database
		#return location of shoes

	def stalk(self):
		#move towards the shoes

	def lostshoe():
		#refind a lost shoe, turn towards location of last view

if __name__ == '__main__':
	
	capture = cv2.VideoCapture(0)
	ret, frame = capture.read()
	cv2.namedWindow('image')
	cv2.imshow("image",frame)

