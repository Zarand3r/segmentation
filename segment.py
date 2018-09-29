import numpy as np
import cv2
import matplotlib.pyplot as plt

def separate(original_image, isPath = True, output_directory = ""):

	# Read the example code down the page on https://stackoverflow.com/questions/11294859/how-to-define-the-markers-for-watershed-in-opencv
	img = original_image
	if (isPath):
		img = cv2.imread(original_image)	
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	# noise removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
	# sure background area
	sure_bg = cv2.dilate(opening,kernel,iterations=3)
	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)
	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)
	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1
	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	markers = cv2.watershed(img,markers)
	img[markers == -1] = [0,0,255]
	cv2.imwrite(output_directory+'/segmented' + '.jpg', img)



def find_contours(original_image, isPath = True,):
	image = original_image
	if(isPath):
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


def simple_segment(original_image, isPath = True, output_directory = ""):
	original = original_image
	if (isPath):
		original = cv2.imread(original_image)		
	edged = cv2.Canny(original, 10, 250)
	(image, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	index = 0
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		index+=1
		if (index < 10):
			new_img=original[y:y+h,x:x+w]
			cv2.imwrite(output_directory + '/' + str(index) + '.jpg', new_img)

def segment(mask_image, original_image, isPath = True, output_directory = ""):
	mask = mask_image
	original = original_image
	if (isPath):
		mask = cv2.imread(mask_image)	
		original = cv2.imread(original_image)		
	edged = cv2.Canny(mask, 10, 250)
	(image, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	index = 0
	for c in contours:
		x,y,w,h = cv2.boundingRect(c)
		if(w>20 and h >20):
			index+=1
			if (index < 50):
				new_img=original[y:y+h,x:x+w]
				cv2.imwrite(output_directory + '/' + str(index) + '.jpg', new_img)

def make_mask(input_file, isPath = True, output_directory = ""):
	input_image = input_file
	original_image = input_file
	if (isPath):
		input_image = cv2.imread(input_file) #Alternatively, load as grayscale with cv2.imread(FILE, 0)
		original_image = cv2.imread(input_file)

	imgray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)#loads in grayscale mode
	# imgray = (255-imgray) #inverts image

	# Otsu's thresholding after Gaussian filtering
	blur = cv2.GaussianBlur(imgray,(5,5),0)
	ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
 

	# # Plot Here
	# plt.figure(figsize=(15,5))
	titles = ['Original','Grayscale', 'Gaussian Blur','Segmentated']
	plt.subplot(1,4,1),plt.imshow(original_image, cmap='Greys_r')
	plt.title(titles[0]), plt.xticks([]), plt.yticks([])
	plt.subplot(1,4,2),plt.imshow(imgray, cmap='Greys_r')
	plt.title(titles[1]), plt.xticks([]), plt.yticks([])
	plt.subplot(1,4,3),plt.imshow(blur, cmap='Greys_r')
	plt.title(titles[2]), plt.xticks([]), plt.yticks([])
	plt.subplot(1,4,4),plt.imshow(thresh, cmap='Greys_r')
	plt.title(titles[3]), plt.xticks([]), plt.yticks([])

	# image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# contour_image = cv2.drawContours(input_image, contours, -1, (0,0,255), 3)
	# plt.figure(figsize=(10,10))
	# plt.subplot(1,2,1),plt.title('Original'),plt.imshow(original_image, cmap='Greys_r')
	# plt.subplot(1,2,2),plt.title('Contours'),plt.imshow(contour_image, cmap='Greys_r')
	plt.show()


	cv2.imwrite(output_directory +'mask' + '.jpg', thresh)
	return thresh 


if __name__ == "__main__":
	INPUT = 'input/nuclear.tif'
	OUTPUT = 'output/nuclei/'
	ORIGINAL = cv2.imread(INPUT)

	mask = make_mask(INPUT, True, OUTPUT)
	# segment(mask, ORIGINAL, False, OUTPUT)

	# separate(original_image = INPUT, output_directory = OUTPUT)


