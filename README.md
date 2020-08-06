# Computer Pointer Controller

The Computer Pointer Controller project demonstrates how to run multiple models in the same machine and coordinate the flow of data between those models. The application uses a person's eye gaze to change the location of the computer's mouse pointer.

## The Pipeline
The flow of data to coordinate will look like this :
![data flow diagram](starter/resources/pipeline.png)

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

### Structure
* starter/src/ contains the application code files 
* resources/models/ contains the models needed to run the app
* /requirements.txt consists of a list of some of the packages and frameworks needed to complete the project

### Software requirements

| Details               |              |
|-----------------------|---------------|
| Programming Language: |  Python 3.5 or 3.6 |
|  Intel® Distribution of OpenVINO™ toolkit   |  2020.1

#### Dependencies
All depencies required by the project can be found in requirements.txt and install using :
`pip install -r requirements.txt`

#### Hardware requirements

* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* OR use of Intel® Neural Compute Stick 2 (NCS2)


### Set Up

Before running the project OpenVINO environment as well as the virtual env for the project must be initialize using :
* Linux: `source venv/bin/activate`
* openvino : `source /opt/intel/openvino/bin/setupvars.sh`

### What model to use
The application uses a [Gaze estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) from the open model zoo to estimate the gaze of the user's eyes and chage the mouse pointer accordingly.The gaze estimation model requires three inputs:
* The head pose
* The left eye image
* The right eye image.
To get get these three inputs we have to use three other OpenVino models:
* [Face Dectection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
These model can be downloaded using the model downloader and placed in resources/models/ directory.

## Demo

To run the app, just fill in [] the appropriate path to the model precision using this command:
`python3 test.py --face [Path to Face Detection model]face-detection-adas-binary-0001 --landmark [Path to Landmarks Dectection model]landmarks-regression-retail-0009 --head [Path to Headpose Estimation model]head-pose-estimation-adas-0001 --gaze [Path to Gaze Estimation model]gaze-estimation-adas-0002 --input_type video --input_file ../bin/demo.mp4 --display True`

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

### Useful links

* [Inference Engine API Docs](https://docs.openvinotoolkit.org/latest/_inference_engine_ie_bridges_python_docs_api_overview.html)
* [Model Documentation](https://docs.openvinotoolkit.org/latest/_models_intel_index.html)
* [PyAutoGUI’s documentation](https://pyautogui.readthedocs.io/en/latest/)

### Command line arguments

| Argument                 | Description                                                  | Required | Default |
| ------------------------ | ------------------------------------------------------------ | :------: | :-----: |
| --face                   | Path to an xml file of face detection model                  |   YES    |   N/A   |
| --landmarks              | Path to an xml file of facial landmarks detection model      |   YES    |   cam   |
| --head                   | Path to an xml file of head pose estimation model            |   YES    |   N/A   |
| --gaze                   | Path to an xml file of gaze estimation model                 |   YES    |   N/A   |
| --device                 | The target device to run inference on                        |   YES    |   CPU   |
| --cpu_extension          | Path CPU extensions library                                  |   NO     |   N/A   |
| --input_type             | Can be 'video' for video file or 'cam' to use webcam         |   YES    |   cam   |
| --input_file             | Path to video file or leave empty for cam                    |    NO    |   N/A   |
| --display                | A flag to display the outputs of intermediate models         |    NO    |  False  |

## Benchmarks

These benchmarks mainly include model loading time and inference time of running model on CPU. It's a tradeoff between the results get from DL Workbench and the log file.

* Face Dectection Model

| Model Precision | Model load time     | Mean inference time  | 
| --------------- | ------------------- | -------------------- | 
| INT8            |                     |         -            | 
| FP16            |                     |         -            | 
| FP32            | around 823.5ms      | 9 ms                 | 

* Landmarks Dectection Model

| Model Precision | Model load time     | Mean inference time  | 
| --------------- | ------------------- | -------------------- | 
| FP16            | around 82.0ms       | 0.28 ms              | 
| FP32            | around 75.0ms       | 0.28 ms              | 
| FP32-INT8       | around 254.8ms      | 0.32 ms              |

* Headpose Estimation Model

| Model Precision | Model load time     | Mean inference time  | 
| --------------- | ------------------- | -------------------- | 
| FP16            | around 133.7ms      | 1.42 ms              | 
| FP32            | around 90.1ms       | 1.42 ms              | 
| FP32-INT8       | around 547.3ms      | 1.14 ms              |

* Gaze Estimation Model

| Model Precision | Model load time     | Mean inference time  | 
| --------------- | ------------------- | -------------------- | 
| FP16            | around 132.7ms      | 1.77 ms              | 
| FP32            | around 122.0ms      | 1.85 ms              | 
| FP32-INT8       | around 631.3ms      | 1.38 ms              |


## Results

From the above we can notice that FP32-INT8 is a good precision to optimize the inference time for all the models. But its downside is the model load time which seems very long in Comparison to FP32 precision. This explains why FP32 is the optimal precision for CPU.

