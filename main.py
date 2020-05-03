import os
try:
	import cv2
except ModuleNotFoundError as e:
	print(e)
	print('--------------------------Downloading Module--------------------')
	os.system('pip install opencv-python')
	import cv2
import numpy as np
import matplotlib.pyplot as plt

import pytesseract
#matplotlib inline
# try:
#     get_ipython().magic("matplotlib inline")
# except:
#     plt.ion()

path = "./Images/car_2.jpg"
image = cv2.imread(path)

def plot_images(img1, img2, title1= '', title2 = ''):
	fig = plt.figure(figsize = [15, 15])
	ax1 = fig.add_subplot(121)
	ax1.imshow(img1, cmap = "gray")
	ax1.set(xticks = [], yticks=[], title = title1)

	ax1 = fig.add_subplot(122)
	ax1.imshow(img2, cmap = "gray")
	ax1.set(xticks = [], yticks=[], title = title2)	
	plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray, 11, 200, 150)
edges = cv2.Canny(blur, 50, 200)
cnts, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#image_copy = image.copy()
#cv2.drawContours(image_copy, cnts, -1, (255, 0, 255), 2)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:40]
#image_cnts_reduced = edges.copy()
#cv2.drawContours(image_cnts_reduced, cnts, -1, (255, 255, 255), 2)

if __name__ == '__main__':
	#cv2.imshow('Edges', edges)
	#cv2.imshow('Blur', blur)
	# k = cv2.waitKey(0)
	# if k == 27:
	# 	cv2.destroyAllWindows()
	#plot_images(image_copy, image_cnts_reduced, title1 = 'Contours', title2 = 'Reduced')
	plate = None
	for c in cnts:
		perimeter = cv2.arcLength(c, True)
		edges_count = cv2.approxPolyDP(c, 0.02* perimeter, True)
		if len(edges_count) == 4:
			x,y,w,h = cv2.boundingRect(c)
			plate = image[y:y+h, x:x+w]
			break
	#cv2.imwrite('plate.png', plate)
	pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
	text = pytesseract.image_to_string(plate, lang = 'eng')
	print(text)
