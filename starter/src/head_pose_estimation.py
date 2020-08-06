'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
import math
import time
from model import Model_X

class HeadPoseEstimation(Model_X):
    '''
    Class for the Face Detection Model.
    '''
    
    def __init__(self, model_name, device='CPU', extensions=None):
        super().__init__(model_name, device='CPU', extensions=None)
    
    def predict(self, image, face_image, face_boxes, display_flag=True):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img = self.preprocess_input(face_image)
        input_name = self.input_name
        input_dict={input_name:input_img}

        start=time.time()
        self.net.start_async(request_id=0, 
            inputs=input_dict)
        status = self.net.requests[0].wait(-1)
        if status == 0:
            inference_time = time.time()- start
            outputs = self.net.requests[0].outputs
            face_image, angles = self.preprocess_output(outputs, image, face_image, face_boxes, display_flag)
            return face_image, angles, inference_time

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        n, c, h, w = self.input_shape
        input_img = image
        input_img=cv2.resize(input_img, (w,h), interpolation = cv2.INTER_AREA)
        input_img = input_img.transpose((2,0,1))
        input_img = input_img.reshape((n, c, h, w))
        return input_img

    def preprocess_output(self, outputs, image, face, facebox, display_flag):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        p = outputs['angle_p_fc'][0][0]
        r = outputs['angle_r_fc'][0][0]
        y = outputs['angle_y_fc'][0][0]


        if display_flag:
            cv2.putText(image,"y:{:.1f}".format(y), (20,20), 0, 0.6, (255,255,0))
            cv2.putText(image,"p:{:.1f}".format(p), (20,40), 0, 0.6, (255,255,0))
            cv2.putText(image,"r:{:.1f}".format(r), (20,60), 0, 0.6, (255,255,0))
                
            xmin, ymin,_ , _ = facebox
            face_center = (xmin + face.shape[1] / 2, ymin + face.shape[0] / 2, 0)
            focal_length = 950.0
            scale = 50
            image = self.draw_axes(image, face_center, y, p, r, scale, focal_length)

        return image, (p, r, y)

    # from : https://knowledge.udacity.com/questions/171017
    def draw_axes(self, frame, center_of_face, yaw, pitch, roll, scale, focal_length):
        yaw *= np.pi / 180.0
        pitch *= np.pi / 180.0
        roll *= np.pi / 180.0
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        Rx = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
        Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
        Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])
        # R = np.dot(Rz, Ry, Rx)
        # ref: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        # R = np.dot(Rz, np.dot(Ry, Rx))
        R = Rz @ Ry @ Rx
        # print(R)
        camera_matrix = self.build_camera_matrix(center_of_face, focal_length)
        xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
        o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        o[2] = camera_matrix[0][0]
        xaxis = np.dot(R, xaxis) + o
        yaxis = np.dot(R, yaxis) + o
        zaxis = np.dot(R, zaxis) + o
        zaxis1 = np.dot(R, zaxis1) + o
        xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
        xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
        xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
        yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
        p1 = (int(xp1), int(yp1))
        xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
        yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
        p2 = (int(xp2), int(yp2))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)
        cv2.circle(frame, p2, 3, (255, 0, 0), 2)
        return frame


    # from : https://knowledge.udacity.com/questions/171017
    def build_camera_matrix(self, center_of_face, focal_length):
        cx = int(center_of_face[0])
        cy = int(center_of_face[1])
        camera_matrix = np.zeros((3, 3), dtype='float32')
        camera_matrix[0][0] = focal_length
        camera_matrix[0][2] = cx
        camera_matrix[1][1] = focal_length
        camera_matrix[1][2] = cy
        camera_matrix[2][2] = 1
        return camera_matrix
