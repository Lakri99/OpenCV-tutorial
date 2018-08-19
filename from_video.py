# necessary imports
import cv2
import numpy as np
import imutils
import argparse

# get the videocamera object
video = cv2.VideoCapture(0)
# read the camera and get each frame
# frame_bool,frame = video.read()
frame_bool = True
while(frame_bool):
    # cv2.imshow("random name",frame)
    frame_bool,frame = video.read()

    # Insert the code to identify faces here
    # construct parser and define the argument
    argument = argparse.ArgumentParser()
    argument.add_argument("-p", "--protext", required = True, help = "protext file path")
    argument.add_argument("-m", "--model", required = True, help = "path to pretrained caffe model")
    argument.add_argument("-c", "--confidence", type = float, default = 0.5, help = "minimum probability to filter weak detections")
    args = vars(argument.parse_args())

    print("reading the model...")

    # read the trained model 
    model = cv2.dnn.readNetFromCaffe(args["protext"], args["model"])

    # read the image and get the dimensions
    frame = imutils.resize(frame, width=400)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    # do preprocessing - image -> blob
    processed_image = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

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
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


    cv2.imshow("random name",frame)
    # Insert way to end the video stream
    key = cv2.waitKey(1) & 0xFF
    if (key == ord("q")):
        break

# blah blah
video.release()
cv2.destroyAllWindows()