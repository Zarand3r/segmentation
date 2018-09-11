import numpy as np
import matplotlib.pyplot as plt
import cv2

def find_contours():
	image = cv2.imread("test.png")
	edged = cv2.Canny(image, 10, 250)
	cv2.imshow("Edges", edged)
	cv2.waitKey(0)
	 
	#applying closing function 
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	cv2.imshow("Closed", closed)
	cv2.waitKey(0)
	 
	#finding_contours 
	(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	 
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
	cv2.imshow("Output", image)
	cv2.waitKey(0)


def simple_segment(original_image, isPath = True):
	original = original_image
	if (isPath):
		original = cv2.imread(original_image)		
	# gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
	edged = cv2.Canny(original, 10, 250)
	(image, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	index = 0
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		index+=1
		if (index < 10):
			new_img=original[y:y+h,x:x+w]
			cv2.imwrite('output/test/'+ str(index) + '.jpg', new_img)

def segment(mask_image, original_image, isPath = True):
	mask = mask_image
	original = original_image
	if (isPath):
		mask = cv2.imread(mask_image)	
		original = cv2.imread(original_image)		
	# gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
	edged = cv2.Canny(mask, 10, 250)
	(image, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	index = 0
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		if(w>20 and h >20):
			index+=1
			if (index < 50):
				new_img=original[y:y+h,x:x+w]
				cv2.imwrite('output/nuclei/'+ str(index) + '.jpg', new_img)

def make_mask():
	FILE = "Nuclear.tif"
	input_image = cv2.imread(FILE) #Alternatively, load as grayscale with cv2.imread(FILE, 0)
	imgray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)#loads in grayscale mode
	# imgray = (255-imgray) #inverts image

	# Otsu's thresholding after Gaussian filtering
	# imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(imgray,(5,5),0)
	ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
 

	# # Plot Here
	# plt.figure(figsize=(15,5))
	titles = ['Original','Grayscale', 'Gaussian Blur','Segmentated']
	plt.subplot(1,4,1),plt.imshow(input_image, cmap='Greys_r')
	plt.title(titles[0]), plt.xticks([]), plt.yticks([])
	plt.subplot(1,4,2),plt.imshow(imgray, cmap='Greys_r')
	plt.title(titles[1]), plt.xticks([]), plt.yticks([])
	plt.subplot(1,4,3),plt.imshow(blur, cmap='Greys_r')
	plt.title(titles[2]), plt.xticks([]), plt.yticks([])
	plt.subplot(1,4,4),plt.imshow(thresh, cmap='Greys_r')
	plt.title(titles[3]), plt.xticks([]), plt.yticks([])

	# image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# contour_image = cv2.drawContours(input_image, contours, -1, (255,255,255), 3)
	# plt.figure(figsize=(10,10))
	# plt.subplot(1,2,1),plt.title('Original Image'),plt.imshow(input_image)#,'red')
	# plt.subplot(1,2,2),plt.title('OpenCV.findContours'),plt.imshow(contour_image,'gray')#,'red')
	plt.show()

	segment(thresh, input_image, False)
	# segment(thresh, imgray, False)


	# cv2.imwrite("output/mask.jpg", thresh)



make_mask()
# simple_segment("test.png")
