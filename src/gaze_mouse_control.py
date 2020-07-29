# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:10:42 2020

@author: Alexey
"""

from face_detection import Model_FaceDetection
from facial_landmarks_detection import Model_FacialLandmarksDetection
from gaze_estimation import Model_GazeEstimation
from head_pose_estimation import Model_HeadPoseEstimation
from mouse_controller import MouseController
from input_feeder import InputFeeder

import cv2
import os
from argparse import ArgumentParser

def build_argparser():
    #Parse command line arguments.

    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help=" Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help="Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str,
                        help="Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or enter cam for webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to detect the face accurately from the video frame.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Target device to run on: "
                             "Options: CPU (default), GPU, FPGA or MYRIAD is acceptable. Sample ")
    parser.add_argument("-v", "--visualizationFlags", required=False, nargs='+',
                            default=[],
                            help="Optional model visualization flags."
                                 "fd = Face Detection, fld = Facial Landmark Detection, hp for Head Pose Estimation, ge for Gaze Estimation"
                                 "Flags should be separated by space." )    
    return parser




args = build_argparser().parse_args()
visualizationFlags = args.visualizationFlags

inputFilePath = args.input
inputFeeder = None
if inputFilePath.lower()=="cam":
        inputFeeder = InputFeeder("cam")
else:
    if not os.path.isfile(inputFilePath):
        print("Unable to find specified video file")
        exit(1)
    inputFeeder = InputFeeder("video",inputFilePath)

modelPathDict = {'FaceDetectionModel':args.facedetectionmodel, 'FacialLandmarksDetectionModel':args.faciallandmarkmodel, 
'GazeEstimationModel':args.gazeestimationmodel, 'HeadPoseEstimationModel':args.headposemodel}

for fileNameKey in modelPathDict.keys():
    if not os.path.isfile(modelPathDict[fileNameKey]):
        print("Unable to find specified "+fileNameKey+" xml file")
        exit(1)
        
face_detection = Model_FaceDetection(modelPathDict['FaceDetectionModel'], args.device, args.cpu_extension)
face_detection.load_model()

face_landmarks_detection = Model_FacialLandmarksDetection(modelPathDict['FacialLandmarksDetectionModel'], args.device, args.cpu_extension)
face_landmarks_detection.load_model()

gaze_estimation = Model_GazeEstimation(modelPathDict['GazeEstimationModel'], args.device, args.cpu_extension)
gaze_estimation.load_model()

head_pose_estimation = Model_HeadPoseEstimation(modelPathDict['HeadPoseEstimationModel'], args.device, args.cpu_extension)
head_pose_estimation.load_model()

inputFeeder.load_data()

mouse_controller = MouseController('medium','fast')

frame_count = 0
for frame in inputFeeder.next_batch():
    frame_count+=1
    if frame_count%5==0:
        cv2.imshow('video',cv2.resize(frame,(500,500)))

    key = cv2.waitKey(60)
    # detecting face
    face, face_coords = face_detection.predict(frame.copy(), args.prob_threshold)
    if type(face)==int:
        print("Face not detected.")
        continue
    
    hp = head_pose_estimation.predict(face.copy())
    # detecting eyes
    leftEye, rightEye, eyeCoords = face_landmarks_detection.predict(face.copy())
    # estimating gaze
    mouseCoord, gazeVector = gaze_estimation.predict(leftEye, rightEye, hp)
    
    if (len(visualizationFlags)>0):
        pFrame = frame.copy()
        if 'fd' in visualizationFlags:
            cv2.rectangle(pFrame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 3)
            pFrame = face
        if 'hp' in visualizationFlags:
            cv2.putText(pFrame, "Head Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp[0],hp[1],hp[2]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if 'fld' in visualizationFlags:
            cv2.rectangle(face, (eyeCoords[0][0]-10, eyeCoords[0][1]-10), (eyeCoords[0][2]+10, eyeCoords[0][3]+10), (0,255,0), 3)
            cv2.rectangle(face, (eyeCoords[1][0]-10, eyeCoords[1][1]-10), (eyeCoords[1][2]+10, eyeCoords[1][3]+10), (0,255,0), 3)
            pFrame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = face
        if 'ge' in visualizationFlags:
            x, y, w = int(gazeVector[0]*12), int(gazeVector[1]*12), 160
            le =cv2.line(leftEye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
            cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
            re = cv2.line(rightEye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
            cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
            face[eyeCoords[0][1]:eyeCoords[0][3],eyeCoords[0][0]:eyeCoords[0][2]] = le
            face[eyeCoords[1][1]:eyeCoords[1][3],eyeCoords[1][0]:eyeCoords[1][2]] = re
            pFrame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = face            
        
        cv2.imshow("model visualization",cv2.resize(pFrame,(500,500)))
 
    
    
    if frame_count%5==0:
        mouse_controller.move(mouseCoord[0],mouseCoord[1])    
    if key==27:
            break
print("Stream ended...")
cv2.destroyAllWindows()
inputFeeder.close()
     
    
