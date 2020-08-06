'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IENetwork, IECore
import cv2
import time
from model import Model_X

class FaceDetection(Model_X):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        super().__init__(model_name, device='CPU', extensions=None)

    def predict(self, image, display_flag=True):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_img = self.preprocess_input(image)
        input_name = self.input_name
        input_dict={input_name:input_img}

        start=time.time()
        self.net.start_async(request_id=0, 
            inputs=input_dict)
        status = self.net.requests[0].wait(-1)
        if status == 0:
            inference_time = time.time()- start
            outputs = self.net.requests[0].outputs[self.output_name]
            coords, image = self.preprocess_output(outputs, image,display_flag)
            return coords, image, inference_time


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

    def preprocess_output(self, outputs, image, display_flag):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        conf_box = []
        for i in range(len(outputs[0][0])):
            box = outputs[0][0][i]
            if box[2] > 0.6:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                conf_box.append([xmin, ymin, xmax, ymax])
                if display_flag:
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
        return conf_box, image
