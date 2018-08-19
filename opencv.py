# imports for the script
import argparse

import cv2
import numpy as np

# construct parser and define the argument
argument = argparse.ArgumentParser()
argument.add_argument("-i", "--image", required = True, help = "path to image file")
argument.add_argument("-p", "--protext", required = True, help = "protext file path")
argument.add_argument("-m", "--model", required = True, help = "path to pretrained caffe model")
argument.add_argument("-c", "--confidence", type = float, default = 0.5, help = "minimum probability to filter weak detections")
args = vars(argument.parse_args())

print("reading the model...")

# read the trained model 
model = cv2.dnn.readNetFromCaffe(args["protext"], args["model"])

# read the image and get the dimensions
image = cv2.imread(args["image"])
(h,w) = image.shape[:2] 

# do preprocessing - image -> blob
processed_image = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

# making predictition for processed image
model.setInput(processed_image)
detections = model.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)