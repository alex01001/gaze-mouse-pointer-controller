
import cv2
import numpy as np
from openvino.inference_engine import IECore

class Model_FaceDetection:
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_weights = self.model_name.split('.')[0]+'.bin'
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
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape
        
    def predict(self, image, prob_threshold):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        
        img_processed = self.preprocess_input(image.copy())
        faces = self.exec_net.infer({self.input_name:img_processed})
        coords = self.preprocess_output(faces, prob_threshold)
        if (len(coords)==0):
            return 0, 0
        # finding first face
        coords = coords[0] 
        h=image.shape[0]
        w=image.shape[1]
        coords = coords* np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        
        face = image[coords[1]:coords[3], coords[0]:coords[2]]
        return face, coords

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

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        resizedImage = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        return np.transpose(np.expand_dims(resizedImage,axis=0), (0,3,1,2))

    def preprocess_output(self, outputs, prob_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        modelOutputs = outputs[self.output_names][0][0]
        coordList =[]
        for output in modelOutputs:
            conf = output[2]
            if conf>prob_threshold:
                xMin=output[3]
                yMin=output[4]
                xMax=output[5]
                yMax=output[6]
                coordList.append([xMin,yMin,xMax,yMax])
        return coordList
        