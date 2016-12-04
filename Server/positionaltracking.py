# USAGE
# python positionaltracking.py
# python positionaltracking.py --display 1
# python positionaltracking.py --display 1 --load 1
# python positionaltracking.py --display 1 --load 1 --fp "/home/pi/calibrated_data.npz"
# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import numpy as np
import os.path
import sys
import threading
import json
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
clients = []
server = None

Mark1_3d = np.float32([0,0,0])
Mark1_Cam1_2d = np.float32([0,0])
Mark1_Cam2_2d = np.float32([0,0])
Mark1_Acquired = False
Mark2_3d = np.float32([0,0,0])
Mark2_Cam1_2d = np.float32([0,0])
Mark2_Cam2_2d = np.float32([0,0])
Mark2_Acquired = False
Mark3_3d = np.float32([0,0,0])
Mark3_Cam1_2d = np.float32([0,0])
Mark3_Cam2_2d = np.float32([0,0])
Mark3_Acquired = False
Mark4_3d = np.float32([0,0,0])
Mark4_Cam1_2d = np.float32([0,0])
Mark4_Cam2_2d = np.float32([0,0])
Mark4_Acquired = False
Mark5_3d = np.float32([0,0,0])
Mark5_Cam1_2d = np.float32([0,0])
Mark5_Cam2_2d = np.float32([0,0])
Mark5_Acquired = False
Mark6_3d = np.float32([0,0,0])
Mark6_Cam1_2d = np.float32([0,0])
Mark6_Cam2_2d = np.float32([0,0])
Mark6_Acquired = False
Mark7_3d = np.float32([0,0,0])
Mark7_Cam1_2d = np.float32([0,0])
Mark7_Cam2_2d = np.float32([0,0])
Mark7_Acquired = False
Mark8_3d = np.float32([0,0,0])
Mark8_Cam1_2d = np.float32([0,0])
Mark8_Cam2_2d = np.float32([0,0])
Mark8_Acquired = False
Mark9_3d = np.float32([0,0,0])
Mark9_Cam1_2d = np.float32([0,0])
Mark9_Cam2_2d = np.float32([0,0])
Mark9_Acquired = False
Mark10_3d = np.float32([0,0,0])
Mark10_Cam1_2d = np.float32([0,0])
Mark10_Cam2_2d = np.float32([0,0])
Mark10_Acquired = False


cam1_projectionMatrix = []
cam1_cameraPosition = []
cam2_projectionMatrix = []
cam2_cameraPosition = []

Outputting_WS = False

def calculate3DPosition(Cam1_2d, Cam2_2d, cam1_Projection_Matrix, cam2_Projection_Matrix):
    """
    Triangulate a 3D position from 2 2D position from camera
    and each camera projection matrix
    :param camera1:
    :param camera2:
    :param Point2D1:
    :param Point2D2:
    :return:
    """

    triangulationOutput = cv2.triangulatePoints(cam1_Projection_Matrix,cam2_Projection_Matrix, Cam1_2d, Cam2_2d)

    mypoint1 = np.array([triangulationOutput[0], triangulationOutput[1], triangulationOutput[2]])
    mypoint1 = mypoint1.reshape(-1, 3)
    mypoint1 = np.array([mypoint1])
    P_24x4 = np.resize(cam1_Projection_Matrix[0], (4,4))
    P_24x4[3,0] = 0
    P_24x4[3,1] = 0
    P_24x4[3,2] = 0
    P_24x4[3,3] = 1

    projected = cv2.perspectiveTransform(mypoint1, P_24x4)
    output = triangulationOutput[:-1]/triangulationOutput[-1]
    #TODO calculate point again with second proj mat, and calculate middle
    return output


def calculateProjectionMatrix(camera_matrix, camera_distortion, listPoint2D, listPoint3D):
    ret, rvec, tvec = cv2.solvePnP(listPoint3D, listPoint2D, camera_matrix, camera_distortion)
    rotM_cam = cv2.Rodrigues(rvec)[0]

    # calculate camera position (= translation), from 0,0,0 point
    cameraPosition = -np.matrix(rotM_cam).T * np.matrix(tvec)
    print ("Camera position : ")
    print (cameraPosition)

    camMatrix = np.append(cv2.Rodrigues(rvec)[0], tvec, 1)
    projectionMatrix = np.dot(camera_matrix, camMatrix)

    return projectionMatrix, cameraPosition

class SimpleWSServer(WebSocket):
    def handleConnected(self):
        clients.append(self)

    def handleClose(self):
        clients.remove(self)

    def handleMessage(self):
	global Mark1_3d
	global Mark1_Cam1_2d
	global Mark1_Cam2_2d
	global Mark1_Acquired
	global Mark2_3d
	global Mark2_Cam1_2d
	global Mark2_Cam2_2d
	global Mark2_Acquired
	global Mark3_3d
	global Mark3_Cam1_2d
	global Mark3_Cam2_2d
	global Mark3_Acquired
	global Mark4_3d
	global Mark4_Cam1_2d
	global Mark4_Cam2_2d
	global Mark4_Acquired
	global Mark5_3d
	global Mark5_Cam1_2d
	global Mark5_Cam2_2d
	global Mark5_Acquired
	global Mark6_3d
	global Mark6_Cam1_2d
	global Mark6_Cam2_2d
	global Mark6_Acquired
	global Mark7_3d
	global Mark7_Cam1_2d
	global Mark7_Cam2_2d
	global Mark7_Acquired
	global Mark8_3d
	global Mark8_Cam1_2d
	global Mark8_Cam2_2d
	global Mark8_Acquired
	global Mark9_3d
	global Mark9_Cam1_2d
	global Mark9_Cam2_2d
	global Mark9_Acquired
	global Mark10_3d
	global Mark10_Cam1_2d
	global Mark10_Cam2_2d
	global Mark10_Acquired
	global mtx
	global dist
	global cam1_projectionMatrix
	global cam1_cameraPosition
	global cam2_projectionMatrix
	global cam2_cameraPosition
	global Outputting_WS
	global LoadSavePath

	MessageArray = json.loads(self.data)
	Answer = {'Answer' : 'Recieved'}
        if MessageArray["Command"] == "Coordinates":
		Cam1coords = Cam1XY
		Cam2coords = Cam2XY
		print("Commmand: Coordinates")
		if MessageArray["Mark"] == "Mark1":
			Mark1_3d = np.float32([MessageArray["Coordinates"]["X"], MessageArray["Coordinates"]["Y"], MessageArray["Coordinates"]["Z"]])
			Mark1_Cam1_2d = np.float32([Cam1coords[0], Cam1coords[1]])
			Mark1_Cam2_2d = np.float32([Cam2coords[0], Cam2coords[1]])
			Mark1_Acquired = True
		if MessageArray["Mark"] == "Mark2":
			Mark2_3d = np.float32([MessageArray["Coordinates"]["X"], MessageArray["Coordinates"]["Y"], MessageArray["Coordinates"]["Z"]])
			Mark2_Cam1_2d = np.float32([Cam1coords[0], Cam1coords[1]])
			Mark2_Cam2_2d = np.float32([Cam2coords[0], Cam2coords[1]])
			Mark2_Acquired = True
		if MessageArray["Mark"] == "Mark3":
			Mark3_3d = np.float32([MessageArray["Coordinates"]["X"], MessageArray["Coordinates"]["Y"], MessageArray["Coordinates"]["Z"]])
			Mark3_Cam1_2d = np.float32([Cam1coords[0], Cam1coords[1]])
			Mark3_Cam2_2d = np.float32([Cam2coords[0], Cam2coords[1]])
			Mark3_Acquired = True
		if MessageArray["Mark"] == "Mark4":
			Mark4_3d = np.float32([MessageArray["Coordinates"]["X"], MessageArray["Coordinates"]["Y"], MessageArray["Coordinates"]["Z"]])
			Mark4_Cam1_2d = np.float32([Cam1coords[0], Cam1coords[1]])
			Mark4_Cam2_2d = np.float32([Cam2coords[0], Cam2coords[1]])
			Mark4_Acquired = True
		if MessageArray["Mark"] == "Mark5":
			Mark5_3d = np.float32([MessageArray["Coordinates"]["X"], MessageArray["Coordinates"]["Y"], MessageArray["Coordinates"]["Z"]])
			Mark5_Cam1_2d = np.float32([Cam1coords[0], Cam1coords[1]])
			Mark5_Cam2_2d = np.float32([Cam2coords[0], Cam2coords[1]])
			Mark5_Acquired = True
		if MessageArray["Mark"] == "Mark6":
			Mark6_3d = np.float32([MessageArray["Coordinates"]["X"], MessageArray["Coordinates"]["Y"], MessageArray["Coordinates"]["Z"]])
			Mark6_Cam1_2d = np.float32([Cam1coords[0], Cam1coords[1]])
			Mark6_Cam2_2d = np.float32([Cam2coords[0], Cam2coords[1]])
			Mark6_Acquired = True
		if MessageArray["Mark"] == "Mark7":
			Mark7_3d = np.float32([MessageArray["Coordinates"]["X"], MessageArray["Coordinates"]["Y"], MessageArray["Coordinates"]["Z"]])
			Mark7_Cam1_2d = np.float32([Cam1coords[0], Cam1coords[1]])
			Mark7_Cam2_2d = np.float32([Cam2coords[0], Cam2coords[1]])
			Mark7_Acquired = True
		if MessageArray["Mark"] == "Mark8":
			Mark8_3d = np.float32([MessageArray["Coordinates"]["X"], MessageArray["Coordinates"]["Y"], MessageArray["Coordinates"]["Z"]])
			Mark8_Cam1_2d = np.float32([Cam1coords[0], Cam1coords[1]])
			Mark8_Cam2_2d = np.float32([Cam2coords[0], Cam2coords[1]])
			Mark8_Acquired = True
		if MessageArray["Mark"] == "Mark9":
			Mark9_3d = np.float32([MessageArray["Coordinates"]["X"], MessageArray["Coordinates"]["Y"], MessageArray["Coordinates"]["Z"]])
			Mark9_Cam1_2d = np.float32([Cam1coords[0], Cam1coords[1]])
			Mark9_Cam2_2d = np.float32([Cam2coords[0], Cam2coords[1]])
			Mark9_Acquired = True
		if MessageArray["Mark"] == "Mark10":
			Mark10_3d = np.float32([MessageArray["Coordinates"]["X"], MessageArray["Coordinates"]["Y"], MessageArray["Coordinates"]["Z"]])
			Mark10_Cam1_2d = np.float32([Cam1coords[0], Cam1coords[1]])
			Mark10_Cam2_2d = np.float32([Cam2coords[0], Cam2coords[1]])
			Mark10_Acquired = True
		AnswerContext = {'Mark_Saved' : MessageArray["Mark"]}
		print ('Mark Recieved')
		print(MessageArray["Mark"])
	if MessageArray["Command"] == "Reset":
		Mark1_3d = np.float32([0,0,0])
		Mark1_Cam1_2d = np.float32([0,0])
		Mark1_Cam2_2d = np.float32([0,0])
		Mark1_Acquired = False
		Mark2_3d = np.float32([0,0,0])
		Mark2_Cam1_2d = np.float32([0,0])
		Mark2_Cam2_2d = np.float32([0,0])
		Mark2_Acquired = False
		Mark3_3d = np.float32([0,0,0])
		Mark3_Cam1_2d = np.float32([0,0])
		Mark3_Cam2_2d = np.float32([0,0])
		Mark3_Acquired = False
		Mark4_3d = np.float32([0,0,0])
		Mark4_Cam1_2d = np.float32([0,0])
		Mark4_Cam2_2d = np.float32([0,0])
		Mark4_Acquired = False
		Mark5_3d = np.float32([0,0,0])
		Mark5_Cam1_2d = np.float32([0,0])
		Mark5_Cam2_2d = np.float32([0,0])
		Mark5_Acquired = False
		Mark6_3d = np.float32([0,0,0])
		Mark6_Cam1_2d = np.float32([0,0])
		Mark6_Cam2_2d = np.float32([0,0])
		Mark6_Acquired = False
		Mark7_3d = np.float32([0,0,0])
		Mark7_Cam1_2d = np.float32([0,0])
		Mark7_Cam2_2d = np.float32([0,0])
		Mark7_Acquired = False
		Mark8_3d = np.float32([0,0,0])
		Mark8_Cam1_2d = np.float32([0,0])
		Mark8_Cam2_2d = np.float32([0,0])
		Mark8_Acquired = False
		Mark9_3d = np.float32([0,0,0])
		Mark9_Cam1_2d = np.float32([0,0])
		Mark9_Cam2_2d = np.float32([0,0])
		Mark9_Acquired = False
		Mark10_3d = np.float32([0,0,0])
		Mark10_Cam1_2d = np.float32([0,0])
		Mark10_Cam2_2d = np.float32([0,0])
		Mark10_Acquired = False
		if os.path.isfile(LoadSavePath):
			os.remove(LoadSavePath)
		AnswerContext = {'Marks Reset' : 'True'}
	if MessageArray["Command"] == "Load":
		if os.path.isfile(LoadSavePath):
			file = np.load(LoadSavePath)
			file.files
			cam1_projectionMatrix = file['cam1_projectionMatrix']
			cam1_cameraPosition = file['cam1_cameraPosition']
			cam2_projectionMatrix = file['cam2_projectionMatrix']
			cam2_cameraPosition = file['cam2_cameraPosition']
			print('Camera Calibrations Loaded')
			AnswerContext = {'Marks Loaded' : 'True'}
		else:
			AnswerContext = {'Error_MSG' : 'No Calibration File Available'}
	if MessageArray["Command"] == "Calibrate":
		AllCoords = True
		if Mark1_Acquired != True:
			AllCoords = False
		if Mark2_Acquired != True:
			AllCoords = False
		if Mark3_Acquired != True:
			AllCoords = False
		if Mark4_Acquired != True:
			AllCoords = False
		if Mark5_Acquired != True:
			AllCoords = False
		if Mark6_Acquired != True:
			AllCoords = False
		if Mark7_Acquired != True:
			AllCoords = False
		if Mark8_Acquired != True:
			AllCoords = False
		if Mark9_Acquired != True:
			AllCoords = False
		if Mark10_Acquired != True:
			AllCoords = False
		if AllCoords == True: 
			print("All Coordinates Collected")
			Mark_All_3d = np.float32([[Mark1_3d[0],Mark1_3d[1],Mark1_3d[2]],[Mark2_3d[0],Mark2_3d[1],Mark2_3d[2]],[Mark3_3d[0],Mark3_3d[1],Mark3_3d[2]],[Mark4_3d[0],Mark4_3d[1],Mark4_3d[2]],[Mark5_3d[0],Mark5_3d[1],Mark5_3d[2]],[Mark6_3d[0],Mark6_3d[1],Mark6_3d[2]],[Mark7_3d[0],Mark7_3d[1],Mark7_3d[2]],[Mark8_3d[0],Mark8_3d[1],Mark8_3d[2]],[Mark9_3d[0],Mark9_3d[1],Mark9_3d[2]],[Mark10_3d[0],Mark10_3d[1],Mark10_3d[2]]])	
			Cam1_All_2d = np.float32([[Mark1_Cam1_2d[0], Mark1_Cam1_2d[1]],[Mark2_Cam1_2d[0], Mark2_Cam1_2d[1]],[Mark3_Cam1_2d[0], Mark3_Cam1_2d[1]],[Mark4_Cam1_2d[0], Mark4_Cam1_2d[1]],[Mark5_Cam1_2d[0], Mark5_Cam1_2d[1]],[Mark6_Cam1_2d[0], Mark6_Cam1_2d[1]],[Mark7_Cam1_2d[0], Mark7_Cam1_2d[1]],[Mark8_Cam1_2d[0], Mark8_Cam1_2d[1]],[Mark9_Cam1_2d[0], Mark9_Cam1_2d[1]],[Mark10_Cam1_2d[0], Mark10_Cam1_2d[1]]])
			Cam2_All_2d = np.float32([[Mark1_Cam2_2d[0], Mark1_Cam2_2d[1]],[Mark2_Cam2_2d[0], Mark2_Cam2_2d[1]],[Mark3_Cam2_2d[0], Mark3_Cam2_2d[1]],[Mark4_Cam2_2d[0], Mark4_Cam2_2d[1]],[Mark5_Cam2_2d[0], Mark5_Cam2_2d[1]],[Mark6_Cam2_2d[0], Mark6_Cam2_2d[1]],[Mark7_Cam2_2d[0], Mark7_Cam2_2d[1]],[Mark8_Cam2_2d[0], Mark8_Cam2_2d[1]],[Mark9_Cam2_2d[0], Mark9_Cam2_2d[1]],[Mark10_Cam2_2d[0], Mark10_Cam2_2d[1]]])
			print(Mark_All_3d)
			print(Cam1_All_2d)
			print(Cam2_All_2d)
			global dist
			global mtx
			cam1_projectionMatrix, cam1_cameraPosition = calculateProjectionMatrix(mtx, dist, Cam1_All_2d, Mark_All_3d)
			print(cam1_projectionMatrix)
			print(cam1_cameraPosition)
			cam2_projectionMatrix, cam2_cameraPosition = calculateProjectionMatrix(mtx, dist, Cam2_All_2d, Mark_All_3d)
			print(cam2_projectionMatrix)
			print(cam2_cameraPosition)

			#Saving the information in a file
			if os.path.isfile(LoadSavePath):
				os.remove(LoadSavePath)
			np.savez(LoadSavePath, cam1_projectionMatrix = cam1_projectionMatrix, cam1_cameraPosition = cam1_cameraPosition, cam2_projectionMatrix = cam2_projectionMatrix, cam2_cameraPosition = cam2_cameraPosition )
			print('camera calibration data saved')
			AnswerContext = {'Cameras Calibrated' : 'True'}
		else:
			print("not enough coords")
			AnswerContext = {'Error_MSG' : 'Not Enough Coordinates'}
		#print("Calibrate")
	if MessageArray["Command"] == "XYZ_ON":
		Outputting_WS = True
		AnswerContext = {'XYZ' : 'ON'}
	if MessageArray["Command"] == "XYZ_OFF":
		Outputting_WS = False
		AnswerContext = {'XYZ' : 'ON'}

	#print(MessageArray)
	Answer.update(AnswerContext)
	json_WS_Send = json.dumps(Answer)
	
	for client in clients:
		client.sendMessage(unicode(json_WS_Send))

def run_server():
    global server
    server = SimpleWebSocketServer('', 85, SimpleWSServer, selectInterval=(1000.0 / 30) / 1000)
    server.serveforever()


t = threading.Thread(target=run_server)
t.start()

class Printer():
    """
    Print things to stdout on one line dynamically
    """
 
    def __init__(self,data):
 
        sys.stdout.write("\r\x1b[K"+data.__str__())
        sys.stdout.flush()

global Cam1XY
global Cam2XY
# focal lengths in pixels
f = 5.5933430628532710e+02

mtx = np.array([[5.5933430628532710e+02, 0., 3.1950000000000000e+02],[0., 5.5933430628532710e+02, 2.3950000000000000e+02],[0., 0., 1.]])
dist = np.array([-1.1983786831888305e-01, 8.7319990419664789e-02, 0., 0.,1.1568198128411766e-01])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description="3D Postional Tracking of the brightest point on two stereo calibrated cameras")
ap.add_argument("-d", "--display", type=int, default=-1, help="Whether or not frames should be displayed")
ap.add_argument("-l", "--load", type=int, default=-1, help="Load Calibration File on startup")
ap.add_argument("-t", "--type", type=int, default=-1, help="1 for color, default brightest point")
ap.add_argument("-f", "--fp", type=str, default='/home/pi/calibrated_data.npz', help="Path to save/load the calibration file")

args = vars(ap.parse_args())
LoadSavePath = args["fp"]

if args["load"] > 0:
	if os.path.isfile(LoadSavePath):
		file = np.load(LoadSavePath)
		file.files
		cam1_projectionMatrix = file['cam1_projectionMatrix']
		cam1_cameraPosition = file['cam1_cameraPosition']
		cam2_projectionMatrix = file['cam2_projectionMatrix']
		cam2_cameraPosition = file['cam2_cameraPosition']
		print('Camera Calibrations Loaded')
		Outputting_WS = True
	else:
		print('Camera Calibration File Invalid or Unavailable')



# created a *threaded *video stream
vs = WebcamVideoStream(src=0).start()
vs2 = WebcamVideoStream(src=1).start()

blueLower = (74, 179, 33)
blueUpper = (112, 255, 196)
redLower = (0, 109, 120)
redUpper = (24, 255, 255)
yellowLower = (0, 146,129)
yellowUpper = (51, 255, 219)

# loop over some frames
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame2 = vs2.read()
	frame = imutils.resize(frame, width=400)
	frame2 = imutils.resize(frame2, width=400)

	if args["type"] > 0:
		hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask1 = cv2.inRange(hsv1, yellowLower, yellowUpper)
		mask1 = cv2.erode(mask1, None, iterations=2)
		mask1 = cv2.dilate(mask1, None, iterations=2)
		cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		center1 = None
		
		Cam1XY = np.float32([0,0])
		Cam2XY = np.float32([0,0])		

		if len(cnts1)>0:
			c1 = max(cnts1, key=cv2.contourArea)
			((x,y), radius) = cv2.minEnclosingCircle(c1)
			M1 = cv2.moments(c1)
			#center1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))
			Cam1XY = np.float32([int(M1["m10"] / M1["m00"]),int(M1["m01"] / M1["m00"])])
		

		hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
		mask2 = cv2.inRange(hsv2, yellowLower, yellowUpper)
		mask2 = cv2.erode(mask2, None, iterations=2)
		mask2 = cv2.dilate(mask2, None, iterations=2)
		cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		center2 = None

		if len(cnts2)>0:
			c2 = max(cnts2, key=cv2.contourArea)
			((x,y), radius) = cv2.minEnclosingCircle(c2)
			M2 = cv2.moments(c2)
			#center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
			Cam2XY = np.float32([int(M2["m10"] / M2["m00"]),int(M2["m01"] / M2["m00"])])
	else:
		#Find the Brightest Point of Cam1
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# apply a Gaussian blur to the image then find the brightest region
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
		cv2.circle(frame, maxLoc, 5, (255, 0, 0), 3)
 
		#Find The Brightest Point of Cam 2
		gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
		# apply a Gaussian blur to the image then find the brightest region
		gray2 = cv2.GaussianBlur(gray2, (3, 3), 0)
		(minVal2, maxVal2, minLoc2, maxLoc2) = cv2.minMaxLoc(gray2)
		cv2.circle(frame2, maxLoc2, 5, (255, 0, 0), 3)
	
	
		Cam1XY = np.float32([maxLoc[0],maxLoc[1]])
		Cam2XY = np.float32([maxLoc2[0],maxLoc2[1]])

	output = " Camera 0: %s , Camera 1: %s " % (Cam1XY, Cam2XY)
	Printer(output)

	if Outputting_WS == True:
		ThreeDPointOutput = []
		ThreeDPointOutput = calculate3DPosition(Cam1XY, Cam2XY, cam1_projectionMatrix, cam2_projectionMatrix)
		XYZCoordinates = { 'X' : str(ThreeDPointOutput[0][0]),   'Y' : str(ThreeDPointOutput[1][0]) , 'Z' : str(ThreeDPointOutput[2][0]) }
		OutputMessageArray = {'MessageType':'Coordinates', 'Coordinates':  XYZCoordinates }
		for client in clients:
        		client.sendMessage(unicode(json.dumps(OutputMessageArray)))
	


	#END NEW STUFF
	# check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		# show the frame to our screen
		cv2.imshow("Frame2", frame2)
		cv2.imshow("Frame", frame)
		#cv2.imshow("Frame2", mask2)
		#cv2.imshow("Frame", mask1)
		key = cv2.waitKey(1) & 0xFF
 
		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break

# do a bit of cleanup
server.close()
cv2.destroyAllWindows()
vs.stop()