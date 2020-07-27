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

Command Line Arguments:

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
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

Performance Analysis of FP32 Precision Models (in seconds )

Name	Loading FP32	FPS ( in seconds )	Avg. Inference time FP32	Total Inference time FP32
Face detection	0.46175241470336914	25.59700246966908	0.03906707440392446	2.304957389831543
Facial Landmark detection	0.9716916084289551	281.5110027120001	0.0035522590249271718	0.20958328247070312
Head Pose detection	0.21136951446533203	312.7201978448859	0.003197746761774612	0.18866705894470215
Gaze Estimation	0.525824785232544	221.20195508454276	0.00452075570316638	0.2667245864868164
Performance Analysis of FP16 Precision Models (in seconds )

Name	Loading FP16	FPS ( in seconds )	Avg. Inference time FP16	Total Inference time FP16
Face detection	--	--	--	--
Facial Landmark detection	1.7589049339294434	284.23515358400977	0.0035182136600300415	0.20757460594177246
Head Pose detection	0.4001595973968506	393.37309921441084	0.0025421158742096463	0.14998483657836914
Gaze Estimation	0.29756903648376465	229.9613478535207	0.0043485568741620595	0.2565648555755615
Results


## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
