# Computer Pointer Controller

This project uses a gaze detection model to control the mouse pointer of a computer. Intel OpenVINO Gaze Estimation model is used to estimate the gaze of the user's eyes and change the mouse pointer position accordingly.  I have used the following pre-trained models from the Model Zoo: face detection model, head-pose estimation model, facial landmarks model, and gaze estimation.


## Project Set Up and Installation (for Windows 10)

- Install OpenVINO™ toolkit and its dependencies to run the application. OpenVINO 2020.4 is used on this project. See the installation documentation here:
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows_fpga.html

- create virtual environment - optional, but recommended 
- clone this repository  
- download the following models from the Model Zoo:
https://download.01.org/opencv/2020/openvinotoolkit/2020.1/open_model_zoo/models_bin/1/

Face Detection Model

python /intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"

Facial Landmark Detection Model

python /intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "landmarks-regression-retail-0009"

HeadPose Estimation Model

python /intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "head-pose-estimation-adas-0001"

Gaze Estimation Model

python /intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "gaze-estimation-adas-0002"



## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
