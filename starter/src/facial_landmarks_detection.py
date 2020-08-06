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

class FacialLandMarksDetection(Model_X):
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        super().__init__(model_name, device='CPU', extensions=None)

    def predict(self, image,  face_image, face_boxes, display_flag=True):
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
            outputs = self.net.requests[0].outputs[self.output_name]
            
            image, left_eye, right_eye = self.preprocess_output(outputs, image, face_boxes, display_flag)
            return image, left_eye, right_eye, inference_time

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

    def preprocess_output(self, outputs, image, facebox, display_flag):
        #outputs = outputs.reshape(1,10)[0]
        width = facebox[2]-facebox[0]
        height = facebox[3]-facebox[1]

        left_x = outputs[0][0] * width
        left_y = outputs[0][1] * height
        right_x = outputs[0][2] * width
        right_y = outputs[0][3] * height
        if display_flag:
            cv2.circle(image, (facebox[0]+left_x, facebox[1]+left_y), 15, (255,0,0), 2)
            cv2.circle(image, (facebox[0]+right_x, facebox[1]+right_y), 15, (255,0,0), 2)

        left_eye_point = [left_x, left_y]
        right_eye_point = [right_x, right_y]
        return image, left_eye_point, right_eye_point

