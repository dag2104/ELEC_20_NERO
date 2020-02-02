
from pyimagesearch.centroidtracker import CentroidTracker
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,default='deploy.prototxt',
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,default='res10_300x300_ssd_iter_140000.caffemodel',
	help="path to Caffe pre-trained model")                                                         
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())

ct = CentroidTracker()
(H, W) = (None, None)
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:


    frame = vs.read()
    frame = imutils.resize(frame, width=400)



    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    rects = []


    for i in range(0, detections.shape[2]):


        if detections[0, 0, i, 2] > args['confidence']:
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype('int'))


            (startX, startY, endX, endY) = box.astype('int')
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,
                          0xFF, 0), 2)


    objects = ct.update(rects)


    for (objectID, centroid) in objects.items():

     
        text = 'ID {}'.format(objectID)
        cv2.putText(
            frame,
            text,
            (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0xFF, 0),
            2,
            )
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0xFF, 0),
                   -1)
        center = (centroid[0], centroid[1])
        pts.appendleft(center)

        for i in np.arange(1, len(pts)):

       
    
            if pts[i - 1] is None or pts[i] is None:
                    continue
    
          
            if counter >= 10 and i == 1 and pts[-10] is not None:
    
                    
    
                dX = pts[-10][0] - pts[i][0]
                dY = pts[-10][1] - pts[i][1]
                (dirX, dirY) = ('', '')
    
                    
                if np.abs(dX) > 20:
                    dirX = ('left' if np.sign(dX) == 1 else 'right')
    
                   
    
                if np.abs(dY) > 20:
                    dirY = ('up' if np.sign(dY) == 1 else 'down')
    
    
                if dirX != '' and dirY != '':
                    direction = '{}-{}'.format(dirY, dirX)
                else:
    
              
    
                    direction = (dirX if dirX != '' else dirY)
    
            thickness = int(np.sqrt(args['buffer'] / float(i + 1)) * 2.5)
          
    
        cv2.putText(
            frame,
            direction,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0xFF),
            3,
            )
        cv2.putText(
            frame,
            'dx: {}, dy: {}'.format(dX, dY),
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0xFF),
            1,
            )



    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    counter += 1

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()