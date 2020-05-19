import numpy as np
import argparse
import cv2
import time
import os
from yolo_utils import parse_data

ap = argparse.ArgumentParser()

ap.add_argument("-t", "--time", default=False,
	help='boolean to toggle displaying of processing time')

ap.add_argument("-i", "--image", default=None,
	help="path to input image")

ap.add_argument("-v", "--video", default=None,
	help="path to input video")

ap.add_argument("-o", "--output", default='output.avi',
	help="path to output video")

ap.add_argument("-y", "--yolo", default='yolo-coco',
	help="base path to YOLO directory")

ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

ap.add_argument("-th", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Mode select
mode = 'Cam'
if args['image'] != None:
	mode = 'Image'
elif args['video'] != None:
	mode = 'Video'


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


if mode == 'Image':
	# check if file exists
	path_to_img = "images/{}".format(args['image'])
	if not os.path.exists(path_to_img):
		print('[INFO] Could not find file at addrees [ {} ],\
		 please retry with correct file name'.format(path_to_img))
		input('\n'+'-> Press enter to exit')
		exit()

	# load our input image and grab its spatial dimensions
	print('[INFO] Image loaded successfully ! ')
	image = cv2.imread(path_to_img)
	(H, W) = image.shape[:2]

	image_to_disp,_,_,_,_ =  parse_data(net, ln, image, args, H, W, COLORS, LABELS)

	# show the output image
	cv2.imshow("Image", image_to_disp)
	cv2.waitKey(0)

if mode == 'Video':
	# check if file exists
	path_to_video = "videos/{}".format(args['video'])
	if not os.path.exists(path_to_video):
		print('[INFO] Could not find file at addrees [ {} ],\
		 please retry with correct file name'.format(path_to_video))
		input('\n'+'-> Press enter to exit')
		exit()

	# initialize the video stream, pointer to output video file, and
	# frame dimensions
	vs = cv2.VideoCapture(path_to_video)
	writer = None
	(W, H) = (None, None)

	# loop over frames from the video file stream
	while True:
		# read the next frame from the file
		(grabbed, f) = vs.read()

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break

		# if the frame dimensions are empty, grab them
		if W is None or H is None:
			(H, W) = f.shape[:2]

		frame,_,_,_,_ = parse_data(net, ln, f, args, H, W, COLORS, LABELS)

		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)

		# write the output frame to disk
		writer.write(frame)

	# release the file pointers
	print("[INFO] cleaning up...")
	writer.release()
	vs.release()

if mode == 'Cam':
	# Infer real-time on webcam

	vid = cv2.VideoCapture(0)
	while True:
		_, f = vid.read()
		H, W = f.shape[:2]

		frame, boxes, confidences, classids, idxs = parse_data(net, ln, f, args,
		 												   H, W, COLORS, LABELS)
		cv2.imshow('webcam', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	vid.release()
	cv2.destroyAllWindows()
