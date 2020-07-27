
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore
import math

class Model_GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_weights = self.model_name.split(".")[0]+'.bin'
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None
        self.plugin = None
        self.network = None
        self.exec_net = None

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        self.network = self.plugin.read_network(model=self.model_name, weights=self.model_weights)
        self.supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        self.unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in self.supported_layers]
        
        if(not self.check_model()):
            exit(1)
                
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

        
    def predict(self, left_eye_image, right_eye_image, hpa):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        outputs = self.exec_net.infer({'head_pose_angles':hpa, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs,hpa)

        return new_mouse_coord, gaze_vector

    def check_model(self):
        # check for unsupported layers
        if len(self.unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers :{}".format(self.unsupported_layers))
            if not self.extensions==None:
                self.plugin.add_extension(self.extensions, self.device)
                self.supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                self.unsupported_layers = [lalyer for lalyer in self.network.layers.keys() if lalyer not in self.supported_layers]
                if len(self.unsupported_layers)!=0:
                    print("unsupported layers found")
                    return False
            else:
                print("cpu extension path not found")
                return False
        return True    

    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        leftEyeResized = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        rightEyeResized = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))

        return np.transpose(np.expand_dims(leftEyeResized,axis=0), (0,3,1,2)), np.transpose(np.expand_dims(rightEyeResized,axis=0), (0,3,1,2))
        
            

    def preprocess_output(self, outputs,hpa):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        
        gazeVector = outputs[self.output_names[0]].tolist()[0]
        rollValue = hpa[2] 
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        x = gazeVector[0] * cosValue + gazeVector[1] * sinValue
        y = -gazeVector[0] *  sinValue+ gazeVector[1] * cosValue
        
        return (x,y), gazeVector
        
