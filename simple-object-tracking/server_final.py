from pyimagesearch.centroidtracker import CentroidTracker
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2 as cv
import io
import socket
import struct
from PIL import Image
import matplotlib.pyplot as pl

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

#for multiprocessing 
import multiprocessing
def string():
    #str
    while True:
        serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        serv.bind(('192.168.43.52', 8080))
        serv.listen(5)
        # conn, addr = serv.accept()
        # from_client = ''
        # data = conn.recv(4096)
        # data = data.decode()
        # data = str(data)
        # #if not data: break
        # from_client += data
        # print(from_client)
        # conn.send("I am SERVER\n".encode())
        
        conn = serv.accept()[0].makefile('rb')
        msg = ""
        msg = conn.read()
        print(msg)
        conn.close()
        print('client disconnected')

#cv
server_socket = socket.socket()
print('Socket created')
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('192.168.43.52', 8000))  # ADD IP HERE
print('Socket bind complete')
server_socket.listen(10)
print('Socket now listening')

# Accept a single connection and make a file-like object out of it
connection = server_socket.accept()[0].makefile('rb')

ct = CentroidTracker()
(H, W) = (None, None)
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""
print("[INFO] loading model...")
net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
try:
    img = None    
    while True:

        # #string multiprocessing
        # process_str = multiprocessing.Process(target = string )
        # process_str.start()
        # process_str.join()

        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
        # if not image_len:
        #     break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        # image = Image.open(image_stream)
        

        # image_stream = io.BytesIO()
        # image_stream.write(connection.read(image_len))
        # image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        # if img is None:
        #     img = pl.imshow(image)
        # else:
        #     img.set_data(image)
        frame = img
        frame = imutils.resize(frame, width=400)



        if W is None or H is None:
            (H, W) = frame.shape[:2]

        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        rects = []


        for i in range(0, detections.shape[2]):


            if detections[0, 0, i, 2] > args['confidence']:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype('int'))


                (startX, startY, endX, endY) = box.astype('int')
                cv.rectangle(frame, (startX, startY), (endX, endY), (0,
                              0xFF, 0), 2)


        objects = ct.update(rects)


        for (objectID, centroid) in objects.items():

         
            text = 'ID {}'.format(objectID)
            cv.putText(
                frame,
                text,
                (centroid[0] - 10, centroid[1] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0xFF, 0),
                2,
                )
            cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 0xFF, 0),
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
                        print(dirX)
        
                       
        
                    if np.abs(dY) > 20:
                        dirY = ('up' if np.sign(dY) == 1 else 'down')
                        print(dirY)
        
                    if dirX != '' and dirY != '':
                        direction = '{}-{}'.format(dirY, dirX)
                    else:
        
                  
        
                        direction = (dirX if dirX != '' else dirY)
        
                thickness = int(np.sqrt(args['buffer'] / float(i + 1)) * 2.5)
              
        
            cv.putText(
                frame,
                direction,
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0xFF),
                3,
                )
            cv.putText(
                frame,
                'dx: {}, dy: {}'.format(dX, dY),
                (10, frame.shape[0] - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0xFF),
                1,
                )



        cv.imshow('Frame', frame)
        key = cv.waitKey(1) & 0xFF
        counter += 1


        #string without multiprocessing
        string()
            
        # if time.time() - start > 10:
        #     break
        if key == ord("q"):
            break
        conn.close()

finally:
    connection.close()
    server_socket.close()