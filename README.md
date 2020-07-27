# Computer Pointer Controller

This project uses a gaze detection model to control the mouse pointer of a computer. Intel OpenVINO Gaze Estimation model is used to estimate the gaze of the user's eyes and change the mouse pointer position accordingly.  I have used the following pre-trained models from the Model Zoo: face detection model, head-pose estimation model, facial landmarks model, and gaze estimation.


## Project Set Up and Installation (for Windows 10)

- Install OpenVINO™ toolkit and its dependencies to run the application. OpenVINO 2020.4 is used on this project. See the installation documentation here:
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows_fpga.html

- create virtual environment - optional, but recommended 
- clone this repository  
- download the following models from the Model Zoo:

  Face Detection Model

  python /intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"

  Facial Landmark Detection Model

  python /intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"

  HeadPose Estimation Model

  python /intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"

  Gaze Estimation Model

  python /intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"
  Model Zoo: https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/



## Demo
To run the project demo, from 'src' folder execute:
python gaze_mouse_control.py -f face-detection-adas-binary-0001.xml -fl landmarks-regression-retail-0009.xml -hp head-pose-estimation-adas-0001.xml -g gaze-estimation-adas-0002.xml -i demo.mp4

If needed, replace the models paths (refer to the documentation section for descriptions of command line argumants)

## Documentation

OpenVINO documentation: https://docs.openvinotoolkit.org/latest/index.html


Application Command Line Arguments:

-h : help

-fl (required) : path to Facial Landmark Detection model's xml file

-f (required) : path to Face Detection model's xml file

-g (required) : path to Gaze Estimation model's xml file

-hp (required) : path to Head Pose Estimation model's xml file

-i (required) : path to input video file OR 'cam' for taking input from webcam

-l (optional) : absolute path of cpu extension if some layers of models are not supported on the device.

-prob (optional) : Probability threshold for model to detect the face accurately from the video frame.

-d (optional) : target device to run the model on. Options are: CPU, GPU, FPGA, MYRIAD.


## Benchmarks

Performance Analysis of FP32 Precision Models (in seconds):

Face detection:
- Loading: 0.46 
- Inference time: 1.8

Facial Landmark detection:
- Loading: 0.97 
- Inference time: 0.21

Head Pose detection:
- Loading: 0.21 
- Inference time: 0.19

Gaze Estimation:
- Loading: 0.52 
- Inference time: 0.27


Performance Analysis of FP16 Precision Models (in seconds):

Face detection: N/A

Facial Landmark detection:
- Loading: 1.7 
- Inference time: 0.21

Head Pose detection:
- Loading: 0.4 
- Inference time: 0.15

Gaze Estimation:
- Loading: 0.3 
- Inference time: 0.26

## Results

FP16 models take slightly more time to load, but take less time for inference. 
