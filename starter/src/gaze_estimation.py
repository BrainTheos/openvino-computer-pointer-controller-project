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


class GazeEstimation(Model_X):
    '''
    Class for the Face Detection Model.
    '''
    
    def __init__(self, model_name, device='CPU', extensions=None):
        super().__init__(model_name, device='CPU', extensions=None)
    
    def predict(self, image, face, fbox, left_eye_image, right_eye_image, hpa, display_flag=True):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_frame, p_left_eye_image,  p_right_eye_image = self.preprocess_input(image, face, left_eye_image, right_eye_image)
        
        #Get multiple input
        net_input = {"head_pose_angles":hpa, "left_eye_image":p_left_eye_image, "right_eye_image":p_right_eye_image}

        start=time.time()
        self.net.start_async(request_id=0, 
            inputs=net_input)
        status = self.net.requests[0].wait(-1)
        if status == 0:
            inference_time = time.time()- start
            outputs = self.net.requests[0].outputs[self.output_name]
            image,gaze_vector = self.preprocess_output(p_frame, left_eye_image, right_eye_image, hpa,outputs, display_flag)
            
            return image, gaze_vector, inference_time

# From : https://github.com/baafw/openvino-eye-gaze-estimation/blob/6aef85b22a495dac6fc50b6e4cbf7a0fd7439ec5/src/gaze_estimation.py#L134
    def preprocess_input(self, frame, face, left_eye_point, right_eye_point):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        lefteye_input_shape =  [1,3,60,60] 
        righteye_input_shape = [1,3,60,60] 

        # crop left eye
        x_center = left_eye_point[0]
        y_center = left_eye_point[1]
        width = lefteye_input_shape[3]
        height = lefteye_input_shape[2]
        # ymin:ymax, xmin:xmax 
        facewidthedge = face.shape[1]
        faceheightedge = face.shape[0]
        
        # check for edges to not crop
        ymin = int(y_center - height//2) if  int(y_center - height//2) >=0 else 0 
        ymax = int(y_center + height//2) if  int(y_center + height//2) <=faceheightedge else faceheightedge

        xmin = int(x_center - width//2) if  int(x_center - width//2) >=0 else 0 
        xmax = int(x_center + width//2) if  int(x_center + width//2) <=facewidthedge else facewidthedge

        print_flag = True
        left_eye_image = face[ymin: ymax, xmin:xmax]
        # print out left eye to frame
        if(print_flag):
            frame[150:150+left_eye_image.shape[0],20:20+left_eye_image.shape[1]] = left_eye_image
        # left eye [1x3x60x60]
        p_frame_left = cv2.resize(left_eye_image, (lefteye_input_shape[3], lefteye_input_shape[2]))
        p_frame_left = p_frame_left.transpose((2,0,1))
        p_frame_left = p_frame_left.reshape(1, *p_frame_left.shape)

        # crop right eye
        x_center = right_eye_point[0]
        y_center = right_eye_point[1]
        width = righteye_input_shape[3]
        height = righteye_input_shape[2]
        # ymin:ymax, xmin:xmax 
        # check for edges to not crop
        ymin = int(y_center - height//2) if  int(y_center - height//2) >=0 else 0 
        ymax = int(y_center + height//2) if  int(y_center + height//2) <=faceheightedge else faceheightedge

        xmin = int(x_center - width//2) if  int(x_center - width//2) >=0 else 0 
        xmax = int(x_center + width//2) if  int(x_center + width//2) <=facewidthedge else facewidthedge

        right_eye_image =  face[ymin: ymax, xmin:xmax]
        # print out left eye to frame
        
        if(print_flag):
            frame[150:150+right_eye_image.shape[0],100:100+right_eye_image.shape[1]] = right_eye_image
            
        # right eye [1x3x60x60]
        p_frame_right = cv2.resize(right_eye_image, (righteye_input_shape[3], righteye_input_shape[2]))
        p_frame_right = p_frame_right.transpose((2,0,1))
        p_frame_right = p_frame_right.reshape(1, *p_frame_right.shape)

        return frame, p_frame_left, p_frame_right



# From : https://knowledge.udacity.com/questions/254779
    def preprocess_output(self,image, left_eye_image, right_eye_image, hpa, outputs, display_flag):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector = outputs[0]
        roll = gaze_vector[2]
        gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        cs = math.cos(roll * math.pi / 180.0)
        sn = math.sin(roll * math.pi / 180.0)
        tmpX = gaze_vector[0] * cs + gaze_vector[1] * sn
        tmpY = -gaze_vector[0] * sn + gaze_vector[1] * cs

        cv2.putText(image,"x:"+str('{:.1f}'.format(tmpX*100))+",y:"+str('{:.1f}'.format(tmpY*100)), (20, 100), 0,0.6, (0,0,255), 1)

        return image, (gaze_vector)
    
    