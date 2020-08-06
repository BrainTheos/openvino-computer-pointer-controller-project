import numpy as np
import time
import os
import argparse
import sys
import cv2
import logging

from input_feeder import InputFeeder
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmarks_detection import FacialLandMarksDetection
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

file_handler = logging.FileHandler('gaze_app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def main(args):
    #model=args.model
    fd_model = args.face
    flmd_model = args.landmarks
    hp_model = args.head
    ge_model = args.gaze
    device=args.device
    display_flag = args.display

    # Init and load models
    fd= FaceDetection(fd_model, device)
    logger.info("######## Model loading Time #######")
    start = time.time()
    fd.load_model()
    logger.info("Face Detection Model: {:.1f}ms".format(1000 * (time.time() - start)) )

    flmd = FacialLandMarksDetection(flmd_model, device)
    start=time.time()
    flmd.load_model()
    logger.info("Facial Landmarks Detection Model: {:.1f}ms".format(1000 * (time.time() - start)) )

    hpe = HeadPoseEstimation(hp_model, device)
    start=time.time()
    hpe.load_model()
    logger.info("HeadPose Estimation Model: {:.1f}ms".format(1000 * (time.time() - start)) )

    ge = GazeEstimation(ge_model, device)
    start=time.time()
    ge.load_model()
    logger.info("Gaze Estimation Model: {:.1f}ms".format(1000 * (time.time() - start)) )

    # Mouse controller
    mc = MouseController("low","fast")

    feed=InputFeeder(input_type=args.input_type, input_file=args.input_file)
    feed.load_data()

    frame_count = 0
    fd_inference_time = 0
    lm_inference_time = 0
    hp_inference_time = 0
    ge_inference_time = 0
    move_mouse = False
    

    for batch in feed.next_batch():
        frame_count += 1
        # Preprocessed output from face detection
        face_boxes, image, fd_time = fd.predict(batch, display_flag)
        fd_inference_time += fd_time

        for face in face_boxes:
            cropped_face = batch[face[1]:face[3],face[0]:face[2]]
            #print(f"Face boxe = {face}")
            # Get preprocessed result from landmarks 
            image,left_eye, right_eye, lm_time = flmd.predict(image, cropped_face, face, display_flag)
            lm_inference_time += lm_time

            # Get preprocessed result from pose estimation
            image,headpose_angels, hp_time = hpe.predict(image, cropped_face, face, display_flag)
            hp_inference_time += hp_time

            # Get preprocessed result from Gaze estimation model
            image, gazevector, ge_time = ge.predict(image, cropped_face, face, left_eye, right_eye, headpose_angels, display_flag)
            #cv2.imshow('Face', cropped_face)
            ge_inference_time += ge_time
            #print(f"Gaze vect {gazevector[0],gazevector[1]}")
            cv2.imshow('img', image)
            if(not move_mouse):
                mc.move(gazevector[0],gazevector[1])
            break

        if cv2.waitKey(1) & 0xFF == ord("k"):
            break
    if(frame_count>0):
        logger.info("###### Models Inference time ######") 
        logger.info(f"Face Detection inference time = {(fd_inference_time*1000)/frame_count} ms")
        logger.info(f"Facial Landmarks Detection inference time = {(lm_inference_time*1000)/frame_count} ms")
        logger.info(f"Headpose Estimation inference time = {(hp_inference_time*1000)/frame_count} ms")
        logger.info(f"Gaze estimation inference time = {(ge_inference_time*1000)/frame_count} ms")
    feed.close()



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--face', required=True, type=str,
                        help="Path to an xml file of face detection model.")
    parser.add_argument('--landmarks', required=True, type=str,
                        help="Path to an xml file of facial landmarks detection model.")
    parser.add_argument('--head', required=True, type=str,
                        help="Path to an xml file of head pose estimation model.")
    parser.add_argument('--gaze', required=True, type=str,
                        help="Path to an xml file of gaze estimation model.")
    parser.add_argument('--device', default='CPU', type=str, help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument('--input_type', type=str, required=True, help="Can be 'video' for video file or 'cam' to use webcam")
    parser.add_argument('--input_file', type=str, required=False, help="Path to video file or leave empty for cam")
    parser.add_argument('--display', default=False, help="Set a flag to display the outputs of intermediate models")
    
    
    
    args=parser.parse_args()

    main(args)


# Command to test the model:
# python3 test.py --face ../../models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 --landmark ../../models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 --head ../../models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 --gaze ../../models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 --input_type cam --display True
